from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)

FEATURE_DIMS = 2048  # 1280 (SD UNet) + 768 (DINOv2-B)


@register_model("Diff3F")
class Diff3FModel(OD3D_Model):
    """Per-frame feature extractor that fuses Stable-Diffusion UNet activations
    with DINOv2 features following the Diff3F recipe.

    forward(frames_gt, frames_pred?) -> (frames_gt, frames_pred)
    Sets frames_pred.featmap (B, F, H, W) and frames_pred.feat (B, F).
    Requires frames_gt.rgb (B, 3, H, W) float [0,1] and optionally frames_gt.depth (B, H, W).
    """

    def __init__(
        self,
        prompt: str = "a photo of an object",
        use_normal_map: bool = False,
        num_images_per_prompt: int = 1,
        num_views: int = 100,
        resolution: int = 512,
        tolerance: float = 0.004,
        freeze: bool = True,
        aggregation_mode: str = "mean",
    ):
        super().__init__()
        self.prompt = prompt
        self.use_normal_map = use_normal_map
        self.num_images_per_prompt = num_images_per_prompt
        self.num_views = num_views
        self.resolution = resolution
        self.tolerance = tolerance
        self.freeze = freeze
        self.aggregation_mode = aggregation_mode
        self.out_dim = FEATURE_DIMS

        # loaded on first forward call to avoid GPU allocation at construction time
        self._pipe = None
        self._dino = None

    def _ensure_models(self, device: torch.device) -> None:
        if self._pipe is not None:
            return
        from o3b.model.diff3f.diff3f.diffusion import init_pipe
        from o3b.model.diff3f.diff3f.dino import init_dino
        logger.info("Diff3F: loading stable-diffusion controlnet pipeline…")
        self._pipe = init_pipe(device, use_normal_map=self.use_normal_map)
        logger.info("Diff3F: loading DINOv2-B model…")
        self._dino = init_dino(device)

    def _forward_object_batch(self, batch):
        """Extract per-vertex diff3f features for a batch sharing a single Mesh."""
        from o3b.model.diff3f.diff3f_utils import compute_features
        from o3b.model.diff3f.diff3f.dataloaders.mesh_container import MeshContainer

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reuse the cached pipeline/DINO model across calls (same as forward()'s
        # frame-based path) instead of rebuilding the whole SD+ControlNet+DINO
        # stack from scratch per mesh — with persistent_workers=True a single
        # worker processes every item in the dataset, so a rebuild-and-free cycle
        # per mesh means repeated from_pretrained()/load_file() staging through
        # host RAM, which is both slow and a source of RAM churn across a run.
        self._ensure_models(device)

        #  [:, [0, 2, 1]]
        source_mesh = MeshContainer(vert=mesh.verts, face=mesh.faces)
        feats = compute_features(
            device, self._pipe, self._dino, source_mesh,
            self.prompt, self.num_views, self.resolution, self.resolution,
            self.tolerance, self.num_images_per_prompt, self.use_normal_map,
            aggregation_mode=self.aggregation_mode,
        )  # (V, FEATURE_DIMS) float cpu, or PerVertexFeatures if aggregation_mode="all_views"

        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1

        if self.aggregation_mode == "all_views":
            # baseline (mean) representation stays on verts3d_feats, unchanged shape/meaning;
            # the new all_views representation lives on parallel fields so nothing downstream
            # that reads verts3d_feats needs to change.
            mean_feats = feats.mean.float()
            # Stay in half precision for all_views: it's documented as half in
            # PerVertexFeatures, and upcasting to float32 here doubled a tensor
            # that was already the single biggest allocation in this call (while
            # the old half copy stayed alive through `feats` a moment longer,
            # briefly needing both at once) — a real contributor to host-RAM
            # OOMs on memory-constrained runtimes.
            all_views_feats = feats.all_views
            all_views_mask  = feats.all_views_mask
            V, K = all_views_feats.shape[0], all_views_feats.shape[1]
            batch.verts3d_feats = mean_feats.unsqueeze(0).expand(B, V, -1).contiguous()
            batch.verts3d_feats_all_views = all_views_feats.unsqueeze(0).expand(B, V, K, -1).contiguous()
            batch.verts3d_feats_all_views_mask = all_views_mask.unsqueeze(0).expand(B, V, K).contiguous()
            return batch

        feats = feats.float()
        V = feats.shape[0]
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch
        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)

        if frames_pred is None:
            frames_pred = frames_gt

        rgb = frames_gt.rgb  # (B, 3, H, W) float [0,1]
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)

        B, _, H, W = rgb.shape
        device = rgb.device
        self._ensure_models(device)

        from o3b.model.diff3f.diff3f.diff3f import arange_pixels
        from o3b.model.diff3f.diff3f.diffusion import add_texture_to_render
        from o3b.model.diff3f.diff3f.dino import get_dino_features

        grid = (
            arange_pixels((H, W), invert_y_axis=False)[0]
            .to(device)
            .reshape(1, H, W, 2)
            .half()
        )

        featmaps: list[torch.Tensor] = []
        for b in range(B):
            img_np = (rgb[b].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)

            if frames_gt.depth is not None:
                depth_map = frames_gt.depth[b].unsqueeze(0).to(device)  # (1, H, W)
            else:
                depth_map = torch.ones(1, H, W, device=device)

            normal_map_input = None
            if (
                self.use_normal_map
                and frames_gt.normal is not None
            ):
                # rgb2normalmap expects (H, W, 1, 3); convert from (3, H, W) float [0,1]
                n = frames_gt.normal[b]  # (3, H, W)
                normal_map_input = n.permute(1, 2, 0).unsqueeze(2).cpu()  # (H, W, 1, 3)

            # diffusion_output[0]: UNet feature tensor (C, H', W')
            # diffusion_output[1]: list of PIL images
            diffusion_output = add_texture_to_render(
                self._pipe,
                img_np,
                depth_map,
                self.prompt,
                normal_map_input=normal_map_input,
                use_latent=False,
                num_images_per_prompt=self.num_images_per_prompt,
                return_image=True,
            )

            # DINOv2 features on the texture-enhanced image
            dino_feats = get_dino_features(
                device, self._dino, diffusion_output[1][0], grid
            )  # (1, 768, H*W)

            # SD UNet intermediate features → upsample → grid-sample → normalise
            with torch.no_grad():
                ft = nn.Upsample(size=(H, W), mode="bilinear")(
                    diffusion_output[0].unsqueeze(0)
                ).to(device)  # (1, C, H, W)
                ft_dim = ft.size(1)
                diff_feats = F.grid_sample(ft, grid, align_corners=False).reshape(
                    1, ft_dim, -1
                )  # (1, C, H*W)
                diff_feats = F.normalize(diff_feats, dim=1)

            combined = torch.hstack([diff_feats * 0.5, dino_feats * 0.5])
            # (1, FEATURE_DIMS, H*W)
            featmaps.append(combined.reshape(1, FEATURE_DIMS, H, W))

        featmap = torch.cat(featmaps, dim=0)  # (B, FEATURE_DIMS, H, W)
        frames_pred.featmap = featmap
        frames_pred.feat = featmap.mean(dim=(-2, -1))  # (B, FEATURE_DIMS)

        return frames_gt, frames_pred

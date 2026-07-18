from __future__ import annotations

import gc
import logging

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


@register_model("TIPSv2")
class TIPSv2Model(OD3D_Model):
    """Self-rendering feature extractor using Google DeepMind's TIPSv2 vision
    encoder (Text-Image Pretraining with Spatial awareness v2).

    Mirrors `Diff3FModel._forward_object_batch` / `SigLIP2Model` but builds a
    `TIPSv2Extractor` instead of loading `pipe`/`dino_model`, and calls
    `compute_features(..., extractor_fn=extractor)` — no Stable Diffusion or
    DINOv2 is loaded, TIPSv2 only ever sees the raw rendered image.

    forward(ObjectBatch) -> ObjectBatch with `.verts3d_feats` set to (B, V, C).
    """

    def __init__(
        self,
        hub_model: str = "google/tipsv2-b14",
        num_views: int = 100,
        resolution: int = 512,
        feature_resolution: int = 448,
        tolerance: float = 0.004,
        freeze: bool = True,
    ):
        super().__init__()
        self.hub_model = hub_model
        self.num_views = num_views
        self.resolution = resolution
        self.feature_resolution = feature_resolution
        self.tolerance = tolerance
        self.freeze = freeze
        # depends on hub_model variant (768/1024/1152/1536); set on first forward
        self.out_dim = None

    def _forward_object_batch(self, batch):
        from o3b.model.diff3f.diff3f_utils import compute_features
        from o3b.model.diff3f.diff3f.dataloaders.mesh_container import MeshContainer
        from o3b.model.tipsv2.extractor import TIPSv2Extractor

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        extractor = TIPSv2Extractor(
            device, hub_model=self.hub_model, resolution=self.feature_resolution
        )
        self.out_dim = extractor.feature_dims

        source_mesh = MeshContainer(vert=mesh.verts, face=mesh.faces)
        feats = compute_features(
            device,
            pipe=None,
            dino_model=None,
            m=source_mesh,
            prompt=None,
            num_views=self.num_views,
            H=self.resolution,
            W=self.resolution,
            tolerance=self.tolerance,
            num_images_per_prompt=1,
            use_normal_map=False,
            extractor_fn=extractor,
        )  # (V, C) float cpu

        del extractor
        torch.cuda.empty_cache()
        gc.collect()

        feats = feats.float()
        V = feats.shape[0]
        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch
        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)
        raise NotImplementedError(
            "TIPSv2Model only supports self-rendering ObjectBatch inputs "
            "(dispatched via mesh.py's _SELF_RENDERING set)."
        )

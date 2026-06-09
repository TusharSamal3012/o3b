from __future__ import annotations

import gc
import logging

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


@register_model("DenseMatcher")
class DenseMatcherModel(OD3D_Model):
    """Per-vertex feature extractor using the full DenseMatcher pipeline.

    Pipeline: SDDINO (2D multi-view) → DiffusionNet (3D geometry) → 512-dim
    normalised per-vertex features.

    forward(ObjectBatch) -> ObjectBatch with verts3d_feats (B, V, width) set.
    Requires the mesh to be present in the ObjectBatch.
    """

    def __init__(
        self,
        pretrained_upsampler_path: str,
        aggre_net_weights_folder: str,
        diffusionnet_ckpt_path: str = "",
        num_views_azimuth: int = 3,
        num_views_elevation: int = 1,
        width: int = 512,
        num_blocks: int = 8,
        use_mv_features: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.pretrained_upsampler_path = pretrained_upsampler_path
        self.aggre_net_weights_folder = aggre_net_weights_folder
        self.diffusionnet_ckpt_path = diffusionnet_ckpt_path
        self.num_views_azimuth = num_views_azimuth
        self.num_views_elevation = num_views_elevation
        self.width = width
        self.num_blocks = num_blocks
        self.use_mv_features = use_mv_features
        self.freeze = freeze
        # out_dim: 768 (raw SDDINO mv_features) or width (DiffusionNet output)
        self.out_dim = 768 if use_mv_features else width
        self._mesh_featurizer = None

    def _ensure_featurizer(self) -> None:
        if self._mesh_featurizer is not None:
            return
        from o3b.model.densematcher.densematcher.meshfeaturizer import MeshFeaturizer
        logger.info("DenseMatcher: loading MeshFeaturizer…")
        featurizer = MeshFeaturizer(
            pretrained_upsampler_path=self.pretrained_upsampler_path,
            num_views=(self.num_views_azimuth, self.num_views_elevation),
            num_blocks=self.num_blocks,
            width=self.width,
            aggre_net_weights_folder=self.aggre_net_weights_folder,
        )
        if self.diffusionnet_ckpt_path:
            logger.info(f"DenseMatcher: loading DiffusionNet weights from {self.diffusionnet_ckpt_path}")
            ckpt = torch.load(self.diffusionnet_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = {
                k.removeprefix("model.extractor_3d."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.extractor_3d")
            }
            featurizer.extractor_3d.load_state_dict(state_dict)
        featurizer.to("cuda")
        featurizer.extractor_2d.featurizer.mem_eff = True
        featurizer.eval()
        self._mesh_featurizer = featurizer

    def _forward_object_batch(self, batch):
        from pytorch3d.structures.meshes import Meshes
        from pytorch3d.renderer.mesh.textures import Textures
        from o3b.model.densematcher.densematcher import diffusion_net

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_featurizer()
        featurizer = self._mesh_featurizer

        verts = mesh.verts.float()  # (V, 3)
        faces = mesh.faces.long()   # (F, 3)
        V = verts.shape[0]

        # Normalise to [-0.15, 0.15] (DenseMatcher expects scale ≈ 0.3 on largest axis)
        center = verts.mean(dim=0)
        verts_c = verts - center
        scale = verts_c.abs().max()
        if scale > 1e-8:
            verts_c = verts_c * (0.15 / scale)

        # Build PyTorch3D Meshes with uniform grey vertex colours
        verts_rgb = torch.full((V, 3), 0.5, dtype=torch.float32)
        textures = Textures(verts_rgb=[verts_rgb.to(device)])
        pt3d_mesh = Meshes(
            verts=[verts_c.to(device)],
            faces=[faces.to(device)],
            textures=textures,
        )

        from o3b.model.densematcher.densematcher.utils import get_uniform_SO3_RT

        # Compute DiffusionNet geometric operators (with vertex normals; sparse → dense)
        normals = pt3d_mesh.verts_normals_list()[0]  # (V, 3)
        operators = diffusion_net.geometry.get_operators(
            verts_c.to(device), faces.to(device),
            k_eig=128, op_cache_dir=None,
            normals=normals,
        )
        frames, mass, L, evals, evecs, gradX, gradY = operators
        operators = (frames, mass, L.to_dense(), evals, evecs, gradX.to_dense(), gradY.to_dense())

        # Compute camera extrinsics around the normalised mesh (mirrors get_mesh)
        bb = pt3d_mesh.get_bounding_boxes()   # [1, 3, 2]
        cam_dist = bb.abs().max() * 2.5       # fixed factor for inference
        Rs, ts, _, _ = get_uniform_SO3_RT(
            num_azimuth=self.num_views_azimuth,
            num_elevation=self.num_views_elevation,
            distance=cam_dist,
            center=bb.mean(2),
            device=device,
        )
        cameras = [Rs, ts]

        with torch.autocast("cuda"):
            with torch.no_grad():
                out_norm, _, mv_features = featurizer(
                    pt3d_mesh, pt3d_mesh, operators, cameras=cameras,
                    return_mvfeatures=True,
                )

        del pt3d_mesh
        torch.cuda.empty_cache()
        gc.collect()

        raw = mv_features if self.use_mv_features else out_norm
        feats = torch.nan_to_num(raw.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch
        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)
        return frames_gt, frames_pred

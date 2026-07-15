from __future__ import annotations

import gc

import torch

from o3b.model.model import OD3D_Model, register_model


@register_model("DINOv3")
class DINOv3Model(OD3D_Model):
    """Self-rendering per-vertex feature extractor using DINOv3 patch tokens."""

    def __init__(
        self,
        hub_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        image_size=384,
        num_views=100,
        resolution=512,
        tolerance=0.004,
        freeze=True,
    ):
        super().__init__()
        self.hub_model = hub_model
        self.image_size = image_size
        self.num_views = num_views
        self.resolution = resolution
        self.tolerance = tolerance
        self.freeze = freeze
        self.out_dim = None

    def _forward_object_batch(self, batch):
        from o3b.model.diff3f.diff3f.dataloaders.mesh_container import (
            MeshContainer,
        )
        from o3b.model.diff3f.diff3f_utils import compute_features
        from o3b.model.dinov3.extractor import DINOv3Extractor

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        extractor = DINOv3Extractor(
            hub_model=self.hub_model,
            image_size=self.image_size,
            device=device,
        )
        self.out_dim = extractor.feature_dims

        source_mesh = MeshContainer(
            vert=mesh.verts,
            face=mesh.faces,
        )

        feats = compute_features(
            device,
            None,
            None,
            source_mesh,
            None,
            self.num_views,
            self.resolution,
            self.resolution,
            self.tolerance,
            1,
            False,
            extractor_fn=extractor,
        )

        del extractor
        torch.cuda.empty_cache()
        gc.collect()

        feats = feats.float()
        vertex_count = feats.shape[0]
        batch_size = (
            batch.verts3d.shape[0]
            if batch.verts3d is not None
            else 1
        )

        batch.verts3d_feats = (
            feats.unsqueeze(0)
            .expand(batch_size, vertex_count, -1)
            .contiguous()
        )

        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch

        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)

        raise NotImplementedError(
            "DINOv3Model only supports self-rendering ObjectBatch inputs"
        )

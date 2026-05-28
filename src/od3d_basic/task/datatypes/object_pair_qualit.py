from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class ObjectPairQualitBatch:
    """Qualitative outputs for a batch of B object pairs."""

    # predicted correspondences: src vertex index for each trgt vertex
    trgt_src_vert_corr:       Optional[Tensor] = None  # (B, V_trgt)   int64
    trgt_src_vert_corr_mask:  Optional[Tensor] = None  # (B, V_trgt)   bool

    # predicted keypoints on the target in the src canonical frame
    trgt_obj_kpts3d_in_src:   Optional[Tensor] = None  # (B, K, 3)

    # rendered visualizations (optional, for logging)
    imgs:                     Optional[Tensor] = None  # (B, 3, H, W)  float32 [0,1]

    # extra per-task qualitative outputs
    extra: dict = field(default_factory=dict)

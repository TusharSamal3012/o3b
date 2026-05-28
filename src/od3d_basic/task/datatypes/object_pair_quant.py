from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class ObjectPairQuantBatch:
    """Quantitative metrics for a batch of B object pairs."""

    # correspondence / keypoint metrics
    kpts_geo_dist:    Optional[Tensor] = None  # (B, K)   geodesic distance per keypoint
    kpts_euc_dist:    Optional[Tensor] = None  # (B, K)   Euclidean distance per keypoint
    kpts_mask:        Optional[Tensor] = None  # (B, K)   bool — valid keypoints

    # aggregated scalars (one value per batch element)
    pck:              Optional[Tensor] = None  # (B,)  percentage of correct keypoints
    pck01:            Optional[Tensor] = None  # (B,)  PCK @ 0.1 * largest target dim
    geo_dist_mean:    Optional[Tensor] = None  # (B,)  mean geodesic distance

    # extra per-task metrics
    extra: dict = field(default_factory=dict)

    def mean(self) -> dict:
        """Return a flat dict of scalar means over the batch dimension."""
        out = {}
        for fname in ("pck", "pck01", "geo_dist_mean"):
            val = getattr(self, fname)
            if val is not None:
                out[fname] = val.mean().item()
        for k, v in self.extra.items():
            if v is not None and hasattr(v, "mean"):
                out[k] = v.mean().item()
        return out

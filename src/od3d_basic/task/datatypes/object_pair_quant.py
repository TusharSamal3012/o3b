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

    def to_wandb_log(self, prefix: str = "batch", wb=None) -> dict:
        """Return a dict ready for wandb.log().

        Scalar means are always included.  Per-keypoint distance tensors are
        logged as wandb.Histogram objects when *wb* is provided.
        """
        out: dict = {f"{prefix}/{k}": v for k, v in self.mean().items()}
        if wb is None:
            return out
        for fname in ("kpts_euc_dist", "kpts_geo_dist"):
            val = getattr(self, fname)
            if val is None:
                continue
            if self.kpts_mask is not None:
                data = val[self.kpts_mask].detach().cpu().float().numpy()
            else:
                data = val.detach().cpu().float().flatten().numpy()
            if data.size:
                out[f"{prefix}/{fname}"] = wb.Histogram(data)
        return out

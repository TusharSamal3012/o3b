from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class FrameObjectPairQuantBatch:
    """Quantitative metrics for a batch of B frame-object pairs.

    Correspondence is evaluated in the **target camera space**: query keypoints
    are transferred into the target frame via the (predicted) relative camera
    pose and compared against the GT target keypoints.
    """

    # ── per-keypoint tensors ──────────────────────────────────────────────────
    cam_kpts_trgt_euc_dist: Optional[Tensor] = None  # (B, K) Euclidean dist (metres)
    kpts_mask:              Optional[Tensor] = None  # (B, K) bool — valid keypoints

    # ── per-sample aggregates ─────────────────────────────────────────────────
    cam_kpts_trgt_euc_dist_mean: Optional[Tensor] = None  # (B,) mean Euclidean dist
    cam_kpts_trgt_pck01:         Optional[Tensor] = None  # (B,) PCK @ 0.1 * trgt_obj_size

    # extra per-task metrics
    extra: dict = field(default_factory=dict)

    def mean(self) -> dict:
        """Flat dict of scalar means over the batch (nan-safe)."""
        import torch as _torch
        out = {}
        for fname in ("cam_kpts_trgt_euc_dist_mean", "cam_kpts_trgt_pck01"):
            val = getattr(self, fname)
            if val is not None:
                finite = val.float()[_torch.isfinite(val.float())]
                if len(finite) > 0:
                    out[fname] = finite.mean().item()
        for k, v in self.extra.items():
            if v is not None and hasattr(v, "mean"):
                out[k] = v.mean().item()
        return out

    def to_wandb_log(self, prefix: str = "batch", wb=None) -> dict:
        out: dict = {f"{prefix}/{k}": v for k, v in self.mean().items()}
        if wb is None or self.cam_kpts_trgt_euc_dist is None:
            return out
        val = self.cam_kpts_trgt_euc_dist
        data = (val[self.kpts_mask] if self.kpts_mask is not None else val.flatten())
        data = data.detach().cpu().float().numpy()
        if data.size:
            out[f"{prefix}/cam_kpts_trgt_euc_dist"] = wb.Histogram(data)
        return out

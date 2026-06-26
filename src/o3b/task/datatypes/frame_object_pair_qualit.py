from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class FrameObjectPairQualitBatch:
    """Qualitative outputs for a batch of B frame-object pairs.

    ``imgs`` is a single composite per sample stacking two rows of
    query|target panels with the predicted correspondences drawn on top:
      * top row    – query|target frame RGB; query keypoints linked to their
        predicted target location, GT target keypoints shown as hollow rings.
      * bottom row – source|target objects rendered from a top-down camera, with
        the same source→target correspondence lines and GT/predicted keypoints.
    """

    imgs: Optional[Tensor] = None  # (B, 3, H, W) float32 [0,1]

    extra: dict = field(default_factory=dict)

    def to_wandb_log(self, prefix: str = "qualit", wb=None, log_imgs: bool = True) -> dict:
        out: dict = {}
        if wb is None or not log_imgs or self.imgs is None:
            return out
        out[f"{prefix}/correspondences"] = [
            wb.Image(img.permute(1, 2, 0).detach().cpu().float().numpy(), caption=f"i{i}")
            for i, img in enumerate(self.imgs)
        ]
        return out

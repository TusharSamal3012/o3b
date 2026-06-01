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

    def to_wandb_log(
        self,
        prefix: str = "qualit",
        wb=None,
        log_imgs: bool = True,
    ) -> dict:
        """Return a dict ready for wandb.log().

        Images in *imgs* are included when *log_imgs* is True and *wb* is
        provided.  Extra tensor fields are skipped (not meaningful in W&B).
        """
        out: dict = {}
        if wb is None or not log_imgs or self.imgs is None:
            return out
        out[f"{prefix}/correspondences"] = [
            wb.Image(
                img.permute(1, 2, 0).detach().cpu().float().numpy(),
                caption=f"i{i}",
            )
            for i, img in enumerate(self.imgs)
        ]
        return out

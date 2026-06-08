from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch import Tensor
from o3b.data.datatypes.frame import Frame


@dataclass(kw_only=True)
class Scene:
    scene_id:      str
    cams_intr4x4:  Optional[Tensor]       = None  # (T, 4, 4)
    rgbs:          Optional[Tensor]       = None  # (T, H, W, 3)
    depths:        Optional[Tensor]       = None  # (T, H, W)
    depths_masks:  Optional[Tensor]       = None  # (T, H, W)
    masks:         Optional[Tensor]       = None  # (T, H, W)  bool
    feats:         Optional[Tensor]       = None  # (T, F)
    featmaps:      Optional[Tensor]       = None  # (T, H, W, F)
    featmaps_lvls: Optional[List[Tensor]] = None  # L x (T, H_l, W_l, F)
    frames:        list[Frame]            = field(default_factory=list)
    events:        Optional[list]         = None  # T elements: str event label or None
    scoreboards:   Optional[list]         = None  # T elements: {"bbox","score_left","score_right"} or None

    @staticmethod
    def from_frames(frames: list[Frame], scene_id: str = "", events: Optional[list] = None) -> Scene:
        def _stack(attr):
            vals = [getattr(f, attr) for f in frames]
            return torch.stack(vals) if all(v is not None for v in vals) else None

        def _stack_lvls(attr):
            per_frame = [getattr(f, attr) for f in frames]
            if any(v is None for v in per_frame):
                return None
            return [
                torch.stack([per_frame[t][l] for t in range(len(per_frame))])
                for l in range(len(per_frame[0]))
            ]

        return Scene(
            scene_id      = scene_id,
            cams_intr4x4  = _stack("cam_intr4x4"),
            rgbs          = _stack("rgb"),
            depths        = _stack("depth"),
            depths_masks  = _stack("depth_mask"),
            masks         = _stack("mask"),
            feats         = _stack("feat"),
            featmaps      = _stack("featmap"),
            featmaps_lvls = _stack_lvls("featmap_lvls"),
            frames        = frames,
            events        = events,
        )

    def viz(
        self,
        show: bool = True,
        max_frames: int = 16,
        ncols: int = 4,
    ) -> Optional[Tensor]:
        """Display frames in an ncols-wide grid with event labels and scoreboard overlay.

        Each cell shows:
          - title: event label (red if event present, grey "none" otherwise)
          - gold bounding box around the scoreboard region (if detected)
          - score text "L : R" drawn just above the bbox (if available)

        Returns (3, H_grid, W_grid) composed image in [0, 1].
        """
        import torch.nn.functional as F

        if self.rgbs is None:
            return None

        rgbs = self.rgbs  # (T, H, W, 3)
        T    = rgbs.shape[0]
        step = max(1, T // max_frames)
        indices = list(range(0, T, step))[:max_frames]

        panels: list[Tensor] = []
        labels: list[str]    = []
        sb_at:  list[Optional[dict]] = []
        for t in indices:
            frame = rgbs[t].float()
            if frame.max() > 1.5:
                frame = frame / 255.0
            frame = frame.clamp(0, 1)

            sb = self.scoreboards[t] if (self.scoreboards and t < len(self.scoreboards)) else None
            frame = _draw_scoreboard_overlay(frame, sb)

            panels.append(frame)
            sb_at.append(sb)
            event = self.events[t] if (self.events and t < len(self.events)) else None
            labels.append(event if event is not None else "none")

        if not panels:
            return None

        # Print scoreboard summary to stdout
        print(f"{self.scene_id}")
        for t, sb in zip(indices, sb_at):
            if sb is None:
                score_str = "no scoreboard data"
            else:
                sl   = sb.get("score_left")
                sr   = sb.get("score_right")
                bbox = sb.get("bbox")
                score_str = (
                    f"score {sl if sl is not None else '?'} : {sr if sr is not None else '?'}"
                    + (f"  bbox {[round(v) for v in bbox]}" if bbox else "  (no bbox)")
                )
            print(f"  frame {t:>5}: {score_str}")

        # Resize all panels to the same height (keep aspect ratio)
        H_out = panels[0].shape[0]
        resized: list[Tensor] = []
        for p in panels:
            h, w = p.shape[:2]
            if h != H_out:
                p = F.interpolate(
                    p.permute(2, 0, 1).unsqueeze(0),
                    size=(H_out, max(1, round(w * H_out / h))),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            resized.append(p)

        # Pad to a full ncols × nrows grid with black frames
        nrows   = (len(resized) + ncols - 1) // ncols
        n_total = nrows * ncols
        blank   = torch.zeros_like(resized[0])
        grid    = resized + [blank] * (n_total - len(resized))

        # Compose grid tensor: (H_grid, W_grid, 3)
        rows_tensors = [
            torch.cat(grid[r * ncols : (r + 1) * ncols], dim=1)
            for r in range(nrows)
        ]
        composed = torch.cat(rows_tensors, dim=0)

        if show:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(max(4, 4 * ncols), max(3, 3 * nrows)),
                squeeze=False,
            )
            flat_axes = axes.flatten()
            for i, ax in enumerate(flat_axes):
                if i < len(resized):
                    ax.imshow(resized[i].cpu().numpy())
                    label = labels[i]
                    ax.set_title(
                        label, fontsize=7,
                        color="red" if label != "none" else "grey",
                        wrap=True,
                    )
                else:
                    ax.axis("off")
                    continue
                ax.axis("off")
            fig.suptitle(self.scene_id, fontsize=10)
            plt.tight_layout()
            plt.show()

        return composed.permute(2, 0, 1)  # (3, H_grid, W_grid)


def _draw_scoreboard_overlay(img: Tensor, sb: Optional[dict]) -> Tensor:
    """Draw a gold bbox + score text on a (H, W, 3) float tensor in [0, 1].

    Returns the annotated tensor (same shape, same dtype).  No-ops when sb is
    None or contains no usable data.
    """
    if sb is None:
        return img

    bbox = sb.get("bbox")
    sl   = sb.get("score_left")
    sr   = sb.get("score_right")

    if bbox is None and sl is None and sr is None:
        return img

    import numpy as np
    from PIL import Image, ImageDraw

    GOLD = (255, 215, 0)
    DARK = (20, 20, 20)

    np_img = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    pil    = Image.fromarray(np_img)
    draw   = ImageDraw.Draw(pil)

    # Bounding box
    if bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        draw.rectangle([x1, y1, x2, y2], outline=GOLD, width=3)
    else:
        x1, y1 = 10, 10

    # Score text just above the bbox
    if sl is not None or sr is not None:
        text = f"{sl if sl is not None else '?'} : {sr if sr is not None else '?'}"
        ty = max(0, y1 - 24)
        bb = draw.textbbox((x1, ty), text)                       # (left,top,right,bottom)
        draw.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=DARK)
        draw.text((x1, ty), text, fill=GOLD)

    return torch.from_numpy(np.array(pil)).float() / 255.0


@dataclass
class SceneBatch:
    """Multiple Scene instances stacked into a batch.

    Variable-length T is handled by padding to T_max; rgbs_mask marks valid frames.
    """
    scene_ids:     list[str]
    rgbs:          Optional[Tensor]  = None  # (B, T_max, H, W, 3)
    rgbs_mask:     Optional[Tensor]  = None  # (B, T_max) bool — True for valid frames
    depths:        Optional[Tensor]  = None  # (B, T_max, H, W)
    depths_masks:  Optional[Tensor]  = None  # (B, T_max, H, W)
    masks:         Optional[Tensor]  = None  # (B, T_max, H, W) bool
    cams_intr4x4:  Optional[Tensor]  = None  # (B, T_max, 4, 4)
    feats:         Optional[Tensor]  = None  # (B, T_max, F)
    events:        Optional[list]    = None  # B x T_max lists of str|None (padded with None)
    scoreboards:   Optional[list]    = None  # B x T_max lists of dict|None (padded with None)


def collate_scenes(scenes: list[Scene]) -> SceneBatch:
    """Collate a list of Scene instances into a SceneBatch.

    Pads T dimension to the maximum scene length across the batch.
    """

    def _pad_stack(attr: str) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Return (padded, mask) for a per-scene tensor field."""
        vals = [getattr(s, attr) for s in scenes]
        if any(v is None for v in vals):
            return None, None
        T_sizes = [v.shape[0] for v in vals]
        T_max = max(T_sizes)
        B = len(vals)
        rest = vals[0].shape[1:]
        out = torch.zeros((B, T_max) + rest, dtype=vals[0].dtype)
        mask = torch.zeros(B, T_max, dtype=torch.bool)
        for i, (v, T) in enumerate(zip(vals, T_sizes)):
            out[i, :T] = v
            mask[i, :T] = True
        uniform = len(set(T_sizes)) == 1
        return out, (None if uniform else mask)

    def _pad_list(attr: str) -> Optional[list]:
        """Pad a per-scene list field to T_max, filling with None."""
        rows = [getattr(s, attr) for s in scenes]
        if all(r is None for r in rows):
            return None
        T_max = max((len(r) for r in rows if r is not None), default=0)
        result = []
        for r in rows:
            row = list(r) if r is not None else []
            row += [None] * (T_max - len(row))
            result.append(row)
        return result

    rgbs, rgbs_mask   = _pad_stack("rgbs")
    depths, _         = _pad_stack("depths")
    depths_masks, _   = _pad_stack("depths_masks")
    masks, _          = _pad_stack("masks")
    cams, _           = _pad_stack("cams_intr4x4")
    feats, _          = _pad_stack("feats")

    return SceneBatch(
        scene_ids    = [s.scene_id for s in scenes],
        rgbs         = rgbs,
        rgbs_mask    = rgbs_mask,
        depths       = depths,
        depths_masks = depths_masks,
        masks        = masks,
        cams_intr4x4 = cams,
        feats        = feats,
        events       = _pad_list("events"),
        scoreboards  = _pad_list("scoreboards"),
    )

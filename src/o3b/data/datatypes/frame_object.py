from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
from o3b.data.datatypes.mesh import Mesh
from o3b.data.datatypes.frame import Frame, _stack_field
from o3b.data.datatypes.object import Object


@dataclass(kw_only=True)
class FrameObject(Frame, Object):
    frame_object_id:  str
    cam_bbox2d:       Optional[Tensor] = None  # (4,)     xyxy pixels
    cam_bbox3d:       Optional[Tensor] = None  # (3,) obj-space side lengths for draw_bbox3d, or (8, 3) cam-space corners
    fo_mask:          Optional[Tensor] = None  # (H, W)   bool  object-instance mask
    obj_kpts2d_mask:  Optional[Tensor] = None  # (K,)     bool  keypoint visible in this frame (occlusion-aware)
    cam_tform4x4_obj:      Optional[Tensor] = None  # (4, 4)  cam←obj SE(3)
    cam_tform4x4_obj_ncds: Optional[Tensor] = None  # (4, 4)  ncds→cam (= cam_tform4x4_obj @ obj_ncds0c_tform4x4_obj)

    def viz(self, show: bool = True) -> Optional[Tensor]:
        """Interactive overlay viewer with CheckButtons to toggle modalities.

        Layers are composited onto a single canvas:
          rgb        – base image
          fo_mask    – green semi-transparent instance mask
          depth      – plasma colourmap overlay
          depth_mask – blue semi-transparent depth-validity mask
          frame_mask – orange semi-transparent frame mask
          cam_bbox2d – yellow 2-D bounding box (draw_bbox)
          cam_bbox3d – 3-D bounding box projected via draw_bbox3d

        Returns the composed (3, H, W) tensor in [0, 1].
        """
        import numpy as np

        layers: dict[str, tuple] = {}

        if self.rgb is not None:
            rgb = self.rgb.float()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            layers["rgb"] = ("rgb", rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy())

        if self.fo_mask is not None:
            m = self.fo_mask.float().cpu().numpy()
            ov = np.zeros((*m.shape, 4), dtype=np.float32)
            ov[..., 1] = 0.9
            ov[..., 3] = m * 0.5
            layers["fo_mask"] = ("overlay", ov)

        if self.depth is not None:
            import matplotlib.cm as _cm
            d = self.depth.float().cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ov = _cm.plasma(d).astype(np.float32)
            ov[..., 3] = 0.6
            layers["depth"] = ("overlay", ov)

        if self.depth_mask is not None:
            dm = self.depth_mask.float().cpu().numpy()
            ov = np.zeros((*dm.shape, 4), dtype=np.float32)
            ov[..., 2] = 0.9
            ov[..., 3] = dm * 0.4
            layers["depth_mask"] = ("overlay", ov)

        if self.mask is not None:
            fm = self.mask.float().cpu().numpy()
            ov = np.zeros((*fm.shape, 4), dtype=np.float32)
            ov[..., 0] = 1.0; ov[..., 1] = 0.55
            ov[..., 3] = fm * 0.4
            layers["frame_mask"] = ("overlay", ov)

        if self.cam_bbox2d is not None:
            layers["cam_bbox2d"] = ("draw_bbox", self.cam_bbox2d.cpu())

        if (self.cam_bbox3d is not None
                and self.cam_intr4x4 is not None
                and self.cam_tform4x4_obj is not None
                and self.cam_bbox3d.shape == torch.Size([3])):
            layers["cam_bbox3d"] = ("draw_bbox3d", (
                self.cam_bbox3d.cpu(),
                self.cam_intr4x4.cpu(),
                self.cam_tform4x4_obj.cpu(),
            ))

        _tform_kpts = self.cam_tform4x4_obj_ncds if self.cam_tform4x4_obj_ncds is not None \
            else self.cam_tform4x4_obj
        if (self.obj_kpts3d is not None
                and self.cam_intr4x4 is not None
                and _tform_kpts is not None):
            layers["obj_kpts3d"] = ("draw_kpts2d", (
                self.obj_kpts3d.cpu(),
                self.obj_kpts3d_mask.cpu() if self.obj_kpts3d_mask is not None else None,
                self.cam_intr4x4.cpu(),
                _tform_kpts.cpu(),
            ))
            if self.obj_kpts2d_mask is not None:
                layers["obj_kpts2d_mask"] = ("draw_kpts2d", (
                    self.obj_kpts3d.cpu(),
                    self.obj_kpts2d_mask.cpu(),
                    self.cam_intr4x4.cpu(),
                    _tform_kpts.cpu(),
                ))

        if not layers:
            return None

        label_order = list(layers.keys())
        active      = {k: True for k in label_order}

        def _hw() -> tuple[int, int]:
            for kind, data in layers.values():
                if kind in ("rgb", "overlay"):
                    return data.shape[:2]
            return 480, 640

        def _compose_numpy() -> np.ndarray:
            H, W = _hw()
            canvas = np.zeros((H, W, 3), dtype=np.float32)
            if "rgb" in layers and active.get("rgb", True):
                canvas = layers["rgb"][1].copy()
            elif "rgb" in layers:
                canvas[:] = 0.15

            for k in label_order:
                if k == "rgb" or not active.get(k, True):
                    continue
                kind, data = layers[k]
                if kind == "overlay":
                    alpha = data[..., 3:4]
                    canvas = canvas * (1 - alpha) + data[..., :3] * alpha

            # apply tensor-based bbox draws
            canvas_t = torch.from_numpy(canvas).permute(2, 0, 1)  # (3,H,W) float [0,1]

            if "cam_bbox2d" in layers and active.get("cam_bbox2d", True):
                from o3b.cv.visual.draw import draw_bbox
                try:
                    canvas_t = draw_bbox(
                        canvas_t, layers["cam_bbox2d"][1],
                        color=(255, 255, 0), line_width=2,
                    ).float().div(255.0)
                except Exception:
                    pass

            if "cam_bbox3d" in layers and active.get("cam_bbox3d", True):
                from o3b.cv.visual.draw import draw_bbox3d
                try:
                    obj_size3d, cam_intr4x4, cam_tform4x4_obj = layers["cam_bbox3d"][1]
                    canvas_t = draw_bbox3d(
                        canvas_t, obj_size3d, cam_intr4x4, cam_tform4x4_obj,
                    ).float().div(255.0)
                except Exception:
                    pass

            for _kpts_key in ("obj_kpts3d", "obj_kpts2d_mask"):
                if _kpts_key in layers and active.get(_kpts_key, True):
                    from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
                    from o3b.data.datatypes.object import _draw_kpts2d_on_imgs
                    try:
                        kpts3d, kpts_mask, cam_intr4x4, cam_tform = layers[_kpts_key][1]
                        H_c, W_c = canvas_t.shape[-2:]
                        kpts2d = proj3d2d_tform4x4_intr4x4_broadcast(
                            pts3d=kpts3d.unsqueeze(0),
                            tform4x4=cam_tform.unsqueeze(0).unsqueeze(0),
                            intr4x4=cam_intr4x4.unsqueeze(0).unsqueeze(0),
                        )  # (1, K, 2)
                        canvas_t = _draw_kpts2d_on_imgs(
                            canvas_t.unsqueeze(0),
                            kpts2d,
                            mask=kpts_mask,
                            radius=max(H_c, W_c) // 50,
                        )[0]
                    except Exception:
                        pass

            return np.clip(canvas_t.permute(1, 2, 0).numpy(), 0, 1)

        if show:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import CheckButtons

            fig, ax = plt.subplots(figsize=(9, 6))
            fig.subplots_adjust(bottom=0.14)
            ax.axis("off")
            cat_str = str(self.category) if self.category is not None else ""
            title   = getattr(self, "frame_id", "")
            if cat_str:
                title = f"{title}  [{cat_str}]"
            ax.set_title(title, fontsize=9)

            im = ax.imshow(_compose_numpy())
            if cat_str:
                ax.text(
                    0.01, 0.97, cat_str,
                    transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    color="white", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
                )

            def _redraw(_label: str = "") -> None:
                im.set_data(_compose_numpy())
                fig.canvas.draw_idle()

            _redraw()

            ax_cb = fig.add_axes([0.05, 0.01, 0.9, 0.10])
            check = CheckButtons(ax_cb, label_order, actives=[True] * len(label_order))

            def on_toggle(label: str) -> None:
                active[label] = not active[label]
                _redraw(label)

            check.on_clicked(on_toggle)
            plt.show()

        canvas = _compose_numpy()
        return torch.from_numpy(canvas).permute(2, 0, 1)


@dataclass
class FrameObjectBatch:
    """Stacked across B samples. Each sample = 1 frame + 1 object."""
    # frame
    cam_intr4x4:      Optional[Tensor]       = None  # (B, 4, 4)
    rgb:              Optional[Tensor]       = None  # (B, 3, H, W)
    depth:            Optional[Tensor]       = None  # (B, H, W)
    depth_mask:       Optional[Tensor]       = None  # (B, H, W)
    frame_mask:       Optional[Tensor]       = None  # (B, H, W)
    feat:             Optional[Tensor]       = None  # (B, F)
    featmap:          Optional[Tensor]       = None  # (B, F, H, W)
    featmap_lvls:     Optional[List[Tensor]] = None  # L x (B, F, H_l, W_l)
    # frame-object
    cam_bbox2d:       Optional[Tensor]       = None  # (B, 4)
    cam_bbox3d:       Optional[Tensor]       = None  # (B, 8, 3)
    fo_mask:          Optional[Tensor]       = None  # (B, H, W)
    obj_kpts2d_mask:  Optional[Tensor]       = None  # (B, K)    bool
    cam_tform4x4_obj: Optional[Tensor]       = None  # (B, 4, 4)
    # object
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[Tensor] = None  # (B,)  int64
    mesh:                    Optional[Mesh]   = None  # shared mesh for all B viewpoints


@dataclass(kw_only=True)
class FrameObjectPair:
    """A query (src) + target (trgt) pair of FrameObjects of the same category."""
    src_object:     FrameObject
    trgt_object:    FrameObject
    src_object_id:  Optional[str] = None
    trgt_object_id: Optional[str] = None


@dataclass
class FrameObjectPairBatch:
    """Stacked across B FrameObjectPair samples (src = query, trgt = target)."""
    # ── keypoints (NCDS object space) ─────────────────────────────────────────
    src_obj_kpts3d:       Optional[Tensor] = None  # (B, K, 3)
    trgt_obj_kpts3d:      Optional[Tensor] = None  # (B, K, 3)
    src_obj_kpts3d_mask:  Optional[Tensor] = None  # (B, K) bool
    trgt_obj_kpts3d_mask: Optional[Tensor] = None  # (B, K) bool
    src_obj_kpts2d_mask:  Optional[Tensor] = None  # (B, K) bool — visible in frame
    trgt_obj_kpts2d_mask: Optional[Tensor] = None  # (B, K) bool
    # ── ground-truth poses ────────────────────────────────────────────────────
    src_cam_tform4x4_obj:       Optional[Tensor] = None  # (B, 4, 4) cam←obj (metric)
    trgt_cam_tform4x4_obj:      Optional[Tensor] = None  # (B, 4, 4)
    src_cam_tform4x4_obj_ncds:  Optional[Tensor] = None  # (B, 4, 4) ncds→cam
    trgt_cam_tform4x4_obj_ncds: Optional[Tensor] = None  # (B, 4, 4)
    src_obj_size:               Optional[Tensor] = None  # (B,) metric max bbox extent
    trgt_obj_size:              Optional[Tensor] = None  # (B,)
    # ── predicted poses (filled by a pose-estimation method; None → use GT) ────
    src_pred_cam_tform4x4_obj:  Optional[Tensor] = None  # (B, 4, 4)
    trgt_pred_cam_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    src_pred_obj_size3d:        Optional[Tensor] = None  # (B, 3) predicted side lengths
    trgt_pred_obj_size3d:       Optional[Tensor] = None  # (B, 3)
    # ── frame inputs (stacked tensors, or per-sample lists if sizes differ) ────
    src_cam_intr4x4:  Optional[Tensor] = None  # (B, 4, 4)
    trgt_cam_intr4x4: Optional[Tensor] = None
    src_rgb:          Optional[object] = None  # (B, 3, H, W)
    trgt_rgb:         Optional[object] = None
    src_depth:        Optional[object] = None  # (B, H, W)
    trgt_depth:       Optional[object] = None
    src_depth_mask:   Optional[object] = None  # (B, H, W) bool
    trgt_depth_mask:  Optional[object] = None
    src_fo_mask:      Optional[object] = None  # (B, H, W) bool object instance mask
    trgt_fo_mask:     Optional[object] = None
    # ── per-sample meshes (for qualitative rendering) + category ───────────────
    src_meshes:   Optional[list]   = None  # list of B Mesh
    trgt_meshes:  Optional[list]   = None  # list of B Mesh
    src_category: Optional[Tensor] = None  # (B,) int64
    trgt_category: Optional[Tensor] = None  # (B,) int64
    # 3-D bounding-box side lengths (obj space) for draw_bbox3d
    src_cam_bbox3d:  Optional[Tensor] = None  # (B, 3)
    trgt_cam_bbox3d: Optional[Tensor] = None  # (B, 3)


def collate_frame_object_pairs(
    samples: list[FrameObjectPair],
    include: Optional[set[str]] = None,
) -> FrameObjectPairBatch:
    def _get(attr, side):
        return _stack_field([getattr(getattr(s, f"{side}_object"), attr, None) for s in samples])

    def _scalar(attr, side):
        return _stack_field([
            torch.as_tensor(getattr(getattr(s, f"{side}_object"), attr), dtype=torch.float32)
            if getattr(getattr(s, f"{side}_object"), attr, None) is not None else None
            for s in samples
        ])

    def _stack_or_list(attr, side):
        vals = [getattr(getattr(s, f"{side}_object"), attr, None) for s in samples]
        if any(v is None for v in vals):
            return None
        return (torch.stack(vals, dim=0)
                if len({tuple(v.shape) for v in vals}) == 1 else vals)

    def _meshes(side):
        ms = [getattr(getattr(s, f"{side}_object"), "mesh", None) for s in samples]
        return ms if any(m is not None for m in ms) else None

    def _cat(side):
        return _stack_field([
            torch.tensor(getattr(getattr(s, f"{side}_object"), "category_id"))
            if getattr(getattr(s, f"{side}_object"), "category_id", None) is not None else None
            for s in samples
        ])

    out = FrameObjectPairBatch()
    for side in ("src", "trgt"):
        setattr(out, f"{side}_obj_kpts3d",            _get("obj_kpts3d", side))
        setattr(out, f"{side}_obj_kpts3d_mask",       _get("obj_kpts3d_mask", side))
        setattr(out, f"{side}_obj_kpts2d_mask",       _get("obj_kpts2d_mask", side))
        setattr(out, f"{side}_cam_tform4x4_obj",      _get("cam_tform4x4_obj", side))
        setattr(out, f"{side}_cam_tform4x4_obj_ncds", _get("cam_tform4x4_obj_ncds", side))
        setattr(out, f"{side}_obj_size",              _scalar("obj_size", side))
        setattr(out, f"{side}_cam_intr4x4",           _get("cam_intr4x4", side))
        setattr(out, f"{side}_rgb",                   _stack_or_list("rgb", side))
        setattr(out, f"{side}_depth",                 _stack_or_list("depth", side))
        setattr(out, f"{side}_depth_mask",            _stack_or_list("depth_mask", side))
        setattr(out, f"{side}_fo_mask",               _stack_or_list("fo_mask", side))
        setattr(out, f"{side}_meshes",                _meshes(side))
        setattr(out, f"{side}_category",              _cat(side))
        setattr(out, f"{side}_cam_bbox3d",            _get("cam_bbox3d", side))
    return out


def collate_frame_objects(
    samples: list[FrameObject],
    include: Optional[set[str]] = None,
) -> FrameObjectBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, attr) for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return FrameObjectBatch(
        cam_intr4x4      = _get("cam_intr4x4"),
        rgb              = _get("rgb"),
        depth            = _get("depth"),
        depth_mask       = _get("depth_mask"),
        frame_mask       = _get("mask"),
        feat             = _get("feat"),
        featmap          = _get("featmap"),
        featmap_lvls     = _get_lvls("featmap_lvls"),
        cam_bbox2d       = _get("cam_bbox2d"),
        cam_bbox3d       = _get("cam_bbox3d"),
        fo_mask          = _get("fo_mask"),
        obj_kpts2d_mask  = _get("obj_kpts2d_mask"),
        cam_tform4x4_obj = _get("cam_tform4x4_obj"),
        pts3d                   = _get("pts3d"),
        pts3d_feats             = _get("pts3d_feats"),
        pts3d_feats_mask        = _get("pts3d_feats_mask"),
        verts3d_feats           = _get("verts3d_feats"),
        verts3d_feats_mask      = _get("verts3d_feats_mask"),
        obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj"),
        obj_kpts3d              = _get("obj_kpts3d"),
        obj_kpts3d_mask         = _get("obj_kpts3d_mask"),
        category = _stack_field([
            torch.tensor(s.category) if s.category is not None else None
            for s in samples
        ]) if (include is None or "category" in include) else None,
    )

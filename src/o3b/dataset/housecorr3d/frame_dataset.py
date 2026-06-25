"""Helper functions for HouseCorr3D frame-object data.

Provides:
  - _index_scene()           : insert one scene's frame-object rows into frames.db
  - modality loaders         : _load_image_tensor, _load_depth_tensor, _load_mask_tensor
  - viser visualization      : _visualize_frame_objects_viser and helpers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch


def _depth_to_pts3d_cam(
    depth: torch.Tensor,       # (H, W) float32 metres
    cam_intr4x4: torch.Tensor, # (4, 4)
    subsample: int = 4,
) -> torch.Tensor:
    """Back-project depth to 3-D points in OpenGL camera space via depth2pts3d_grid.

    Back-projects at full resolution so intrinsics remain correct, then subsamples.
    Returns (N, 3) float32 tensor.
    """
    from o3b.cv.geometry.transform import depth2pts3d_grid

    d = depth.float()
    pts3d = depth2pts3d_grid(d[None, None], cam_intr4x4[None].float(), opengl=True)[0]  # 3×H×W

    pts3d_sub = pts3d[:, ::subsample, ::subsample]  # 3×H'×W'
    valid     = (d[::subsample, ::subsample] > 0).view(-1)
    if not valid.any():
        return torch.zeros(0, 3)
    return pts3d_sub.reshape(3, -1).T[valid]  # (N, 3)


def _add_frustum_to_scene(
    server,
    cam_intr4x4: torch.Tensor,  # (4, 4)
    H: int,
    W: int,
    name: str = "/frame/frustum",
    color: tuple = (200, 200, 50),
    scale: float = 0.3,
    world_tform4x4_cam: "Optional[torch.Tensor]" = None,
) -> list:
    """Draw camera frustum wireframe by back-projecting the four image corners.

    Back-projection (OpenGL: +Y up, -Z forward):
      pixel (u, v) → X = (u-cx)/fx, Y = -(v-cy)/fy, Z = -1

    This correctly handles off-center principal points (e.g. after a crop
    transform where cx/cy shift outside the image bounds).
    """
    import numpy as np

    fx = cam_intr4x4[0, 0].item()
    fy = cam_intr4x4[1, 1].item()
    cx = cam_intr4x4[0, 2].item()
    cy = cam_intr4x4[1, 2].item()

    d = scale  # frustum depth in camera space
    # TL, TR, BR, BL corners at depth d (Z = -d in OpenGL)
    corners = np.array([
        [-cx / fx * d,         cy / fy * d,          -d],  # TL (u=0, v=0)
        [(W - cx) / fx * d,    cy / fy * d,          -d],  # TR (u=W, v=0)
        [(W - cx) / fx * d,   -(H - cy) / fy * d,    -d],  # BR (u=W, v=H)
        [-cx / fx * d,        -(H - cy) / fy * d,    -d],  # BL (u=0, v=H)
    ], dtype=np.float32)

    origin = np.zeros(3, dtype=np.float32)
    edges = np.array([
        [origin,       corners[0]],  # O → TL
        [origin,       corners[1]],  # O → TR
        [origin,       corners[2]],  # O → BR
        [origin,       corners[3]],  # O → BL
        [corners[0],   corners[1]],  # TL → TR
        [corners[1],   corners[2]],  # TR → BR
        [corners[2],   corners[3]],  # BR → BL
        [corners[3],   corners[0]],  # BL → TL
    ], dtype=np.float32)  # (8, 2, 3)

    # Points are in OpenGL camera space; apply world_tform4x4_cam directly.
    # No 180° convention flip needed (unlike add_camera_frustum which expects OpenCV).
    if world_tform4x4_cam is not None:
        wxyz = _rot3x3_to_wxyz(world_tform4x4_cam[:3, :3].float())
        pos  = tuple(float(v) for v in world_tform4x4_cam[:3, 3].float().cpu())
    else:
        wxyz = (1.0, 0.0, 0.0, 0.0)  # identity
        pos  = (0.0, 0.0, 0.0)

    handles = []
    try:
        h = server.scene.add_line_segments(
            name,
            points=edges,
            colors=np.array(color, dtype=np.uint8),
            line_width=1.5,
            wxyz=wxyz,
            position=pos,
        )
        handles.append(h)
    except Exception:
        pass
    return handles


def _add_rgb_image_to_scene(
    server,
    rgb: torch.Tensor,          # (3, H, W) float32 [0, 1]
    cam_intr4x4: torch.Tensor,  # (4, 4)
    depth: Optional[torch.Tensor],  # (H, W) or None
    name: str = "/frame/rgb",
    world_tform4x4_cam: "Optional[torch.Tensor]" = None,
) -> Optional[object]:
    """Add the RGB image as a flat panel at median scene depth.

    If world_tform4x4_cam is None the panel is placed in camera space (cam-centric).
    Otherwise the panel centre and orientation are mapped into world/object space.
    """
    from o3b.cv.geometry.transform import depth2pts3d_grid

    H_img, W_img = rgb.shape[1], rgb.shape[2]
    fx = cam_intr4x4[0, 0].item()
    fy = cam_intr4x4[1, 1].item()
    cx = cam_intr4x4[0, 2].item()
    cy = cam_intr4x4[1, 2].item()

    if depth is not None:
        pts3d = depth2pts3d_grid(depth.float()[None, None], cam_intr4x4[None].float(), opengl=True)[0]  # 3×H×W
        valid = depth > 0
        d_place = float(-pts3d[2][valid].median()) if valid.any() else 1.0
        # panel centre: back-project image-centre pixel at d_place
        Hh, Wh = H_img // 2, W_img // 2
        d_ctr = float(depth[Hh, Wh])
        if d_ctr > 0:
            cx_cam = float(pts3d[0, Hh, Wh]) / d_ctr * d_place
            cy_cam = float(pts3d[1, Hh, Wh]) / d_ctr * d_place
        else:
            cx_cam =  (Wh - cx) / fx * d_place
            cy_cam = -(Hh - cy) / fy * d_place
    else:
        d_place = 1.0
        cx_cam =  (W_img / 2.0 - cx) / fx * d_place
        cy_cam = -(H_img / 2.0 - cy) / fy * d_place

    render_w = W_img * d_place / fx
    render_h = H_img * d_place / fy

    if world_tform4x4_cam is not None:
        W = world_tform4x4_cam.float()
        cam_pt  = torch.tensor([cx_cam, cy_cam, -d_place, 1.0])
        pos     = tuple(float(v) for v in (W @ cam_pt)[:3].cpu())
        wxyz    = _rot3x3_to_wxyz(W[:3, :3])
    else:
        pos  = (float(cx_cam), float(cy_cam), float(-d_place))
        wxyz = (1.0, 0.0, 0.0, 0.0)

    # flip image vertically so pixel rows go bottom→top in 3D (matching camera Y↑)
    img_np = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[::-1].copy()
    try:
        h = server.scene.add_image(
            name,
            image=img_np,
            render_width=float(render_w),
            render_height=float(render_h),
            wxyz=wxyz,
            position=pos,
        )
        return h
    except Exception:
        return None


def _rot3x3_to_wxyz(R: torch.Tensor) -> tuple:
    """Convert a 3×3 rotation matrix to a wxyz unit quaternion."""
    m = R.cpu().numpy().astype(float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        w, x = 0.25 / s, (m[2, 1] - m[1, 2]) * s
        y, z = (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _add_axes_to_scene(
    server,
    name: str,
    tform4x4: torch.Tensor,   # (4, 4) with possible isotropic scale in [:3, :3]
    axes_length: float = 0.1,
    axes_radius: float = 0.005,
    labels: tuple = ("right", "top", "back"),  # X, Y, Z tip labels
) -> list:
    """Add a coordinate frame (X=red, Y=green, Z=blue) with axis-tip text labels.

    The rotation block may carry an isotropic scale; SVD strips it before
    computing the quaternion.  Returns a list of viser handles.
    """
    U, _, Vh = torch.linalg.svd(tform4x4[:3, :3].float())
    wxyz     = _rot3x3_to_wxyz(U @ Vh)
    position = tuple(float(v) for v in tform4x4[:3, 3].float().cpu())

    result = []
    try:
        h = server.scene.add_frame(
            name, wxyz=wxyz, position=position,
            axes_length=axes_length, axes_radius=axes_radius,
        )
        result.append(h)
    except Exception:
        pass

    # Labels are children of the frame node, so positions are in the frame's LOCAL
    # coordinate system — viser applies the parent transform automatically.
    _colors  = [(220, 50, 50), (50, 200, 50), (50, 50, 220)]  # R, G, B
    _keys    = ("x", "y", "z")
    _offsets = [(axes_length, 0.0, 0.0), (0.0, axes_length, 0.0), (0.0, 0.0, axes_length)]
    for key, text, color, local_pos in zip(_keys, labels, _colors, _offsets):
        try:
            h = server.scene.add_label(
                f"{name}/lbl_{key}", text, position=local_pos, color=color,
            )
            result.append(h)
        except TypeError:
            try:
                h = server.scene.add_label(f"{name}/lbl_{key}", text, position=local_pos)
                result.append(h)
            except Exception:
                pass
        except Exception:
            pass

    return result


def _fo_to_obj_centric(fo) -> "tuple":
    """Return (fo_obj_centric, world_tform4x4_cam) for object-centric visualization.

    Applies a pure translation so the object centre lands at the world origin.
    Camera-space orientation and metric scale are preserved (no rotation, no NCDS
    normalisation scale):
      - world_tform4x4_cam  = [I | -t_obj]  (pure translation by -object_centre_in_cam)
      - fo_obj.cam_tform4x4_obj_ncds = world_tform4x4_cam @ original (mesh correct in world)

    Pass world_tform4x4_cam to the _add_*_to_scene helpers.
    Returns (fo, None) unchanged when cam_tform4x4_obj_ncds is not available.
    """
    from dataclasses import replace as _dc_replace

    if fo.cam_tform4x4_obj_ncds is None:
        return fo, None

    # Object centre in camera space is the translation column of cam_tform4x4_obj_ncds
    t_obj = fo.cam_tform4x4_obj_ncds[:3, 3].float()

    world_tform4x4_cam = torch.eye(4)
    world_tform4x4_cam[:3, 3] = -t_obj

    world_tform4x4_obj_ncds = world_tform4x4_cam @ fo.cam_tform4x4_obj_ncds.float()

    fo_obj = _dc_replace(fo, cam_tform4x4_obj_ncds=world_tform4x4_obj_ncds)
    return fo_obj, world_tform4x4_cam


def _add_depth_pc_to_scene(
    server,
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor],
    cam_intr4x4: torch.Tensor,
    name: str = "/frame/depth_pc",
    subsample: int = 4,
    world_tform4x4_cam: "Optional[torch.Tensor]" = None,
) -> Optional[object]:
    """Back-project depth to camera space and add as a coloured point cloud.

    If world_tform4x4_cam is given the points are further transformed into
    world/object space before being sent to viser.
    """
    import numpy as np

    pts = _depth_to_pts3d_cam(depth, cam_intr4x4, subsample=subsample)
    if pts.shape[0] == 0:
        return None
    
    if world_tform4x4_cam is not None:
        W    = world_tform4x4_cam.float()
        pts_h = torch.cat([pts, torch.ones(pts.shape[0], 1)], dim=-1)  # (N, 4)
        pts   = (W @ pts_h.T).T[:, :3]

    pts_np = pts.cpu().numpy()

    if rgb is not None:
        H, W = depth.shape
        d_sub = depth[::subsample, ::subsample]
        valid = (d_sub > 0).view(-1)
        ys = torch.arange(0, H, subsample)[:d_sub.shape[0]]
        xs = torch.arange(0, W, subsample)[:d_sub.shape[1]]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        y_idx = grid_y.reshape(-1)[valid].long().clamp(0, H - 1)
        x_idx = grid_x.reshape(-1)[valid].long().clamp(0, W - 1)
        colors_np = rgb[:, y_idx, x_idx].permute(1, 0).cpu().numpy()  # (N, 3)
    else:
        colors_np = np.full((pts_np.shape[0], 3), 0.6, dtype=np.float32)

    try:
        h = server.scene.add_point_cloud(
            name,
            points=pts_np,
            colors=colors_np,
            point_size=0.003,
        )
        return h
    except Exception:
        return None


def _visualize_frame_objects_viser(dataset, debug: bool = False, obj_centric: bool = False) -> None:
    """Interactive viser browser for HouseCorr3D frame-object items.

    obj_centric=False (default): camera-centric — camera at origin, object transformed.
    obj_centric=True:            object-centric — object mesh at origin, camera placed
                                 at inv(cam_tform4x4_obj_ncds) in NCDS/world space.
    """
    import time

    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser")
        return

    import numpy as np

    server = viser.ViserServer()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    n = len(dataset._frame_rows_id)
    idx     = [0]
    handles: list = []
    _img_handle = [None]   # GuiImageHandle for the sidebar modality view
    _mod_dd     = [None]   # GuiDropdown handle
    _mod_imgs   = [{}]     # current dict[str, uint8 HxWx3]

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _build_sidebar_imgs(fo) -> "dict":
        imgs = {}
        if fo.rgb is not None:
            rgb_np = (fo.rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imgs["rgb"] = rgb_np
            if fo.cam_bbox2d is not None:
                x1, y1, x2, y2 = (int(v) for v in fo.cam_bbox2d.cpu().tolist())
                bbox_np = rgb_np.copy()
                H_b, W_b = bbox_np.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W_b - 1, x2), min(H_b - 1, y2)
                t = max(2, H_b // 200)  # line thickness
                color = (255, 220, 0)
                bbox_np[y1:y1+t, x1:x2+1] = color   # top
                bbox_np[y2-t:y2, x1:x2+1] = color   # bottom
                bbox_np[y1:y2+1, x1:x1+t] = color   # left
                bbox_np[y1:y2+1, x2-t:x2] = color   # right
                imgs["cam_bbox2d"] = bbox_np
        if fo.depth is not None:
            d = fo.depth.cpu().numpy()
            valid = d > 0
            d_vis = np.zeros_like(d)
            if valid.any():
                d_vis[valid] = d[valid] / d[valid].max()
            imgs["depth"] = (np.stack([d_vis] * 3, axis=-1) * 255).astype(np.uint8)
        if fo.fo_mask is not None:
            m = fo.fo_mask
            if m.dim() == 3:
                m = m[0]
            imgs["mask"] = (
                np.stack([m.float().cpu().numpy()] * 3, axis=-1) * 255
            ).astype(np.uint8)
        _tform_for_kpts = fo.cam_tform4x4_obj_ncds if fo.cam_tform4x4_obj_ncds is not None \
            else fo.cam_tform4x4_obj
        if (
            fo.obj_kpts3d is not None
            and fo.cam_intr4x4 is not None
            and _tform_for_kpts is not None
            and fo.rgb is not None
        ):
            try:
                from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
                from o3b.data.datatypes.object import _draw_kpts2d_on_imgs
                import torch as _torch
                H_k, W_k = rgb_np.shape[:2]
                base_t = _torch.from_numpy(rgb_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                kpts2d = proj3d2d_tform4x4_intr4x4_broadcast(
                    pts3d=fo.obj_kpts3d.float().cpu().unsqueeze(0),
                    tform4x4=_tform_for_kpts.float().cpu().unsqueeze(0).unsqueeze(0),
                    intr4x4=fo.cam_intr4x4.float().cpu().unsqueeze(0).unsqueeze(0),
                )  # (1, K, 2)
                drawn = _draw_kpts2d_on_imgs(
                    base_t, kpts2d,
                    mask=fo.obj_kpts3d_mask.cpu() if fo.obj_kpts3d_mask is not None else None,
                    radius=max(H_k, W_k) // 50,
                )
                imgs["obj_kpts3d"] = (drawn[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            except Exception:
                pass
        return imgs

    def _update_sidebar_img() -> None:
        if _mod_dd[0] is None or _img_handle[0] is None:
            return
        mod = _mod_dd[0].value
        imgs = _mod_imgs[0]
        if mod in imgs:
            _img_handle[0].image = imgs[mod]

    def _load(i: int) -> None:
        _clear()
        fo = dataset._load_frame_object(i)
        if dataset._transform is not None:
            fo = dataset._transform(fo)

        H_img = fo.rgb.shape[1] if fo.rgb is not None else 480
        W_img = fo.rgb.shape[2] if fo.rgb is not None else 640

        # Sidebar images use the original camera-space tform for 2-D projection;
        # build them now before _fo_to_obj_centric replaces cam_tform4x4_obj_ncds.
        fo_for_sidebar = fo

        # Object-centric: invert the cam←obj transform so the mesh stays at the origin
        # and the camera is placed in object/NCDS space.
        world_tform4x4_cam = None
        if obj_centric:
            fo, world_tform4x4_cam = _fo_to_obj_centric(fo)

        # Camera coordinate axes
        cam_pose = world_tform4x4_cam if world_tform4x4_cam is not None else torch.eye(4)
        handles.extend(_add_axes_to_scene(server, "/camera/axes", cam_pose,
                                          axes_length=0.15, axes_radius=0.006))

        # Mesh: in obj-centric mode fo.cam_tform4x4_obj_ncds == identity → mesh at origin
        if fo.mesh is not None and fo.cam_tform4x4_obj_ncds is not None:
            fo_world = fo.transform(fo.cam_tform4x4_obj_ncds)
            hs = fo_world._build_scene_handles(server, "/object", (0.0, 0.0, 0.0))
            handles.extend(h for h in hs.values() if h is not None)

        # Object coordinate axes
        if fo.cam_tform4x4_obj_ncds is not None:
            ax_len = max(0.05, float(fo.obj_size or 0.2) * 0.75)
            handles.extend(_add_axes_to_scene(server, "/object/axes", fo.cam_tform4x4_obj_ncds,
                                              axes_length=ax_len, axes_radius=ax_len * 0.04))

        if fo.cam_intr4x4 is not None:
            handles.extend(_add_frustum_to_scene(
                server, fo.cam_intr4x4, H=H_img, W=W_img,
                world_tform4x4_cam=world_tform4x4_cam,
            ))
            if fo.rgb is not None:
                h = _add_rgb_image_to_scene(
                    server, fo.rgb, fo.cam_intr4x4, depth=fo.depth,
                    world_tform4x4_cam=world_tform4x4_cam,
                )
                if h is not None:
                    handles.append(h)

            if fo.depth is not None:
                h = _add_depth_pc_to_scene(
                    server, fo.depth, fo.rgb, fo.cam_intr4x4,
                    world_tform4x4_cam=world_tform4x4_cam,
                )
                if h is not None:
                    handles.append(h)

        # ── sidebar modality images ───────────────────────────────────────────
        imgs = _build_sidebar_imgs(fo_for_sidebar)
        _mod_imgs[0] = imgs
        if imgs:
            keys = list(imgs.keys())
            if _mod_dd[0] is None:
                _mod_dd[0] = server.gui.add_dropdown(
                    "Modality", options=keys, initial_value=keys[0]
                )

                @_mod_dd[0].on_update
                def _(_e):
                    _update_sidebar_img()
            else:
                _mod_dd[0].options = keys
                if _mod_dd[0].value not in keys:
                    _mod_dd[0].value = keys[0]

            mod = _mod_dd[0].value if _mod_dd[0].value in imgs else keys[0]
            if _img_handle[0] is None:
                _img_handle[0] = server.gui.add_image(imgs[mod], label="Frame")
            else:
                _img_handle[0].image = imgs[mod]

        row = dataset._frame_rows[dataset._frame_rows_id[i]]
        cat = row.get("category", "")
        fid = row.get("frame_id", str(i))
        label.value = f"[{i + 1}/{n}]  {fid}  cat={cat}"
        print(f"  [{i + 1}/{n}] {fid}  cat={cat}")

    with server.gui.add_folder("Navigation"):
        label    = server.gui.add_text("Item",   initial_value="loading…")
        btn_prev = server.gui.add_button("← Prev")
        btn_next = server.gui.add_button("Next →")

    @btn_prev.on_click
    def _(_):
        idx[0] = (idx[0] - 1) % n
        _load(idx[0])

    @btn_next.on_click
    def _(_):
        idx[0] = (idx[0] + 1) % n
        _load(idx[0])

    _load(0)
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Use Prev / Next in the panel to browse. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")


# ── indexing helpers ──────────────────────────────────────────────────────────

def _index_scene(
    cur,
    scene_dir: Path,
    scene_name: str,
    split: str,
    data_type: str,
    path_raw: Path,
    kpts_preprocess: Path,
    limit: Optional[int] = None,
    filter_kpts: bool = False,
) -> tuple[int, int]:
    """Insert frame-object rows for one scene. Returns (n_total, n_matching).

    n_matching counts rows that satisfy the index-time filter:
    - filter_kpts=True  → only rows with has_kpts=1
    - filter_kpts=False → same as n_total

    Stops early once *limit* matching rows have been inserted (None = no limit).
    """
    from o3b.dataset.housecorr3d._frame_utils import (
        build_cam_intr4x4,
        build_cam_tform4x4_obj,
        build_obj_cam_tform,
        _png_size,
    )

    frame_ids_color = sorted(
        p.stem[: -len("_color")]
        for p in scene_dir.iterdir()
        if p.name.endswith("_color.png")
    )

    # Read image dimensions once for the scene (all frames share the same resolution).
    scene_img_size: tuple[int, int] | None = None
    for fid in frame_ids_color:
        p = scene_dir / f"{fid}_color.png"
        if p.exists():
            try:
                scene_img_size = _png_size(p)
            except Exception:
                pass
            break

    n       = 0  # total rows inserted
    n_match = 0  # rows matching the filter
    for frame_id_raw in frame_ids_color:
        meta_path = scene_dir / f"{frame_id_raw}_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        cam_meta   = meta.get("camera", {})
        intrinsics = cam_meta.get("intrinsics", {})

        try:
            cam_intr4x4_list   = build_cam_intr4x4(intrinsics, img_size=scene_img_size)
            cam_tform4x4_world = build_cam_tform4x4_obj(cam_meta)
        except Exception:
            cam_intr4x4_list   = None
            cam_tform4x4_world = None

        rgb_path   = scene_dir / f"{frame_id_raw}_color.png"
        depth_path = scene_dir / f"{frame_id_raw}_depth.exr"
        mask_path  = scene_dir / f"{frame_id_raw}_mask.exr"

        rgb_rpath   = str(rgb_path.relative_to(path_raw))   if rgb_path.exists()   else None
        depth_rpath = str(depth_path.relative_to(path_raw)) if depth_path.exists() else None
        mask_rpath  = str(mask_path.relative_to(path_raw))  if mask_path.exists()  else None

        for obj_idx, (obj_name, obj) in enumerate(meta.get("objects", {}).items()):
            is_valid  = int(bool(obj.get("is_valid", True)))
            category  = obj.get("meta", {}).get("class_name")
            object_id = obj.get("meta", {}).get("oid") or obj_name
            # Mask EXR pixel value = integer prefix of the object key (e.g. "5_mango_..." → 5),
            # NOT the sequential 'id' field (1, 2, 3...).
            try:
                mask_id = int(obj_name.split("_")[0])
            except (ValueError, IndexError):
                mask_id = None
            bbox_side_len = obj.get("meta", {}).get("bbox_side_len")  # [w, h, d] metres
            scale_raw = obj.get("meta", {}).get("scale", None)
            if isinstance(scale_raw, (list, tuple)):
                obj_scale = float(scale_raw[0])   # isotropic
            elif scale_raw is not None:
                obj_scale = float(scale_raw)
            else:
                obj_scale = 1.0
            has_kpts = 1 if (kpts_preprocess / object_id / "kpts3d.pt").exists() else 0

            # per-object cam_tform4x4_obj: uses the object's own quaternion/translation
            try:
                cam_tform4x4_obj_list = build_obj_cam_tform(obj)
            except Exception:
                cam_tform4x4_obj_list = cam_tform4x4_world

            frame_id = f"{data_type}/{scene_name}/{frame_id_raw}/{obj_idx}"

            cur.execute(
                """
                INSERT OR IGNORE INTO frames
                    (frame_id, scene_name, object_idx, mask_id, split, data_type,
                     category, object_id,
                     rgb_path, depth_path, mask_path,
                     cam_intr4x4, cam_tform4x4_obj, obj_size3d,
                     obj_scale, has_kpts, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id, scene_name, obj_idx, mask_id, split, data_type,
                    category, object_id,
                    rgb_rpath, depth_rpath, mask_rpath,
                    json.dumps(cam_intr4x4_list)      if cam_intr4x4_list      is not None else None,
                    json.dumps(cam_tform4x4_obj_list) if cam_tform4x4_obj_list is not None else None,
                    json.dumps(bbox_side_len)          if bbox_side_len         is not None else None,
                    obj_scale, has_kpts, is_valid,
                ),
            )
            n += 1
            if not filter_kpts or has_kpts:
                n_match += 1
            if limit is not None and n_match >= limit:
                return n, n_match
    return n, n_match


# ── modality loaders ─────────────────────────────────────────────────────────

def _load_image_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load PNG/JPEG → (3, H, W) float32 in [0, 1]."""
    if not path.exists():
        return None
    try:
        import torchvision.io as tio
        img = tio.read_image(str(path))          # (C, H, W) uint8
        return img.float() / 255.0
    except Exception:
        return None


def _load_depth_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load depth EXR → (H, W) float32 in metres."""
    if not path.exists():
        return None
    try:
        import os, cv2, numpy as np
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return torch.from_numpy(arr.astype(np.float32))
    except Exception:
        return None


def _load_mask_tensor(path: Path, mask_id: int) -> Optional[torch.Tensor]:
    """Load scene mask EXR and return a bool (H, W) for the given object mask_id.

    Omni6DPose stores all objects in one EXR.  Channel 2 (BGR) scaled by 255
    gives an integer object-id per pixel matching the 'id' field in meta.json.
    """
    if not path.exists():
        return None
    try:
        import os, cv2, numpy as np
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        if arr is None:
            return None
        ids = np.array(arr[:, :, 2] * 255, dtype=np.uint8)
        ids[ids == 255] = 0  # bug fix for test_real subset (spurious 255 values)
        return torch.from_numpy(ids == mask_id)
    except Exception:
        return None

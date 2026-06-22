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
    """Back-project depth to 3-D points in camera space.

    OpenGL convention: −Z_cam forward, depth positive for visible points → Z < 0.
    Returns (N, 3) float32 tensor (subsampled by `subsample`).
    """
    H, W = depth.shape
    fx = cam_intr4x4[0, 0].item()
    fy = cam_intr4x4[1, 1].item()
    cx = cam_intr4x4[0, 2].item()
    cy = cam_intr4x4[1, 2].item()

    d = depth.float()[::subsample, ::subsample]  # (H', W')
    valid = d > 0
    if not valid.any():
        return torch.zeros(0, 3)

    Hp, Wp = d.shape
    ys = torch.arange(0, H, subsample, dtype=torch.float32)[:Hp]
    xs = torch.arange(0, W, subsample, dtype=torch.float32)[:Wp]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    X =  (grid_x - cx) / fx * d
    Y = -(grid_y - cy) / fy * d  # image Y↓ → camera Y↑
    Z = -d  # OpenGL: camera looks in −Z, so visible points have Z < 0
    pts = torch.stack([X, Y, Z], dim=-1).view(-1, 3)
    return pts[valid.view(-1)]


def _add_frustum_to_scene(
    server,
    cam_intr4x4: torch.Tensor,  # (4, 4)
    H: int,
    W: int,
    name: str = "/frame/frustum",
    color: tuple = (200, 200, 50),
    scale: float = 0.3,
) -> list:
    """Add a camera frustum at the origin of the camera-space viser scene.

    OpenGL convention: camera looks in −Z, so the frustum is rotated 180° around Y
    so that viser's local +Z (the frustum-opening direction) maps to world −Z.
    """
    import math

    fx  = cam_intr4x4[0, 0].item()
    fov = 2.0 * math.atan(W / (2.0 * fx))

    handles = []
    try:
        h = server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=W / H,
            scale=scale,
            wxyz=(0.0, 0.0, 1.0, 0.0),  # 180° around Y: local +Z → world −Z
            position=(0.0, 0.0, 0.0),
            color=color,
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
) -> Optional[object]:
    """Add the RGB image as a flat panel in camera space at median scene depth."""
    H_img, W_img = rgb.shape[1], rgb.shape[2]
    fx = cam_intr4x4[0, 0].item()
    fy = cam_intr4x4[1, 1].item()
    cx = cam_intr4x4[0, 2].item()
    cy = cam_intr4x4[1, 2].item()

    if depth is not None:
        d_vals = depth[depth > 0]
        d_place = float(d_vals.median()) if d_vals.numel() > 0 else 1.0
    else:
        d_place = 1.0

    render_w = W_img * d_place / fx
    render_h = H_img * d_place / fy

    # image-plane centre in camera space (principal-point offset, Y flipped: image↓ → cam↑)
    cx_cam =  (W_img / 2.0 - cx) / fx * d_place
    cy_cam = -(H_img / 2.0 - cy) / fy * d_place

    # flip image vertically so pixel rows go bottom→top in 3D (matching camera Y↑)
    img_np = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[::-1].copy()
    try:
        h = server.scene.add_image(
            name,
            image=img_np,
            render_width=float(render_w),
            render_height=float(render_h),
            wxyz=(1.0, 0.0, 0.0, 0.0),  # identity; panel at −Z is visible from +Z (camera origin)
            position=(float(cx_cam), float(cy_cam), float(-d_place)),
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


def _add_depth_pc_to_scene(
    server,
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor],
    cam_intr4x4: torch.Tensor,
    name: str = "/frame/depth_pc",
    subsample: int = 4,
) -> Optional[object]:
    """Back-project depth to camera space and add as a coloured point cloud."""
    import numpy as np

    pts = _depth_to_pts3d_cam(depth, cam_intr4x4, subsample=subsample)
    if pts.shape[0] == 0:
        return None

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


def _visualize_frame_objects_viser(dataset, debug: bool = False) -> None:
    """Interactive viser browser for HouseCorr3D frame-object items.

    The scene is in camera space (camera at origin, +Z forward).
    Each frame shows:
      • object mesh transformed to camera space via cam_tform4x4_obj_ncds
        (NCDS → cam; incorporates both rotation and per-object metric scale)
      • camera frustum at the origin
      • RGB image panel at median depth
      • depth point cloud coloured by RGB
    """
    import time

    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser")
        return

    server = viser.ViserServer()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    n = len(dataset._frame_rows_id)
    idx     = [0]
    handles: list = []

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _load(i: int) -> None:
        _clear()
        # Load with all modalities (dataset cfg already has modalities=None for viz)
        fo = dataset._load_frame_object(i)

        H_img = fo.rgb.shape[1] if fo.rgb is not None else 480
        W_img = fo.rgb.shape[2] if fo.rgb is not None else 640

        # Camera coordinate axes at origin (X=red, Y=green, Z=blue)
        handles.extend(_add_axes_to_scene(server, "/camera/axes", torch.eye(4),
                                          axes_length=0.15, axes_radius=0.006))

        # Mesh in camera space: fo.mesh is in NCDS; cam_tform4x4_obj_ncds maps NCDS→cam
        if fo.mesh is not None and fo.cam_tform4x4_obj_ncds is not None:
            fo_cam = fo.transform(fo.cam_tform4x4_obj_ncds)
            hs = fo_cam._build_scene_handles(server, "/object", (0.0, 0.0, 0.0))
            handles.extend(h for h in hs.values() if h is not None)

        # Object coordinate axes: position = object center in cam space, orientation = R_obj
        if fo.cam_tform4x4_obj_ncds is not None:
            ax_len = max(0.05, float(fo.obj_size or 0.2) * 0.75)
            handles.extend(_add_axes_to_scene(server, "/object/axes", fo.cam_tform4x4_obj_ncds,
                                              axes_length=ax_len, axes_radius=ax_len * 0.04))

        if fo.cam_intr4x4 is not None:
            handles.extend(_add_frustum_to_scene(server, fo.cam_intr4x4, H=H_img, W=W_img))

            if fo.rgb is not None:
                h = _add_rgb_image_to_scene(server, fo.rgb, fo.cam_intr4x4, depth=fo.depth)
                if h is not None:
                    handles.append(h)

            if fo.depth is not None:
                h = _add_depth_pc_to_scene(server, fo.depth, fo.rgb, fo.cam_intr4x4)
                if h is not None:
                    handles.append(h)

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

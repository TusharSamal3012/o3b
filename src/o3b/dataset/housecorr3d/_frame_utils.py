"""Camera math helpers for HouseCorr3DFrame indexing.

Mirrors the logic from od3d.od3d_datasets.omni6dpose.dataset.extract_meta
but using only torch / standard Python — no od3d dependency required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def build_cam_intr4x4(
    intrinsics: dict,
    rgb_path: Optional[Path] = None,
    img_size: Optional[tuple] = None,
) -> list:
    """Build 4×4 intrinsic matrix, adjusting for any image downsampling.

    Omni6DPose stores intrinsics at the sensor resolution, but the
    saved color PNG may have been downsampled.  We correct fx/fy/cx/cy
    by the actual downsample ratio.

    Pass ``img_size=(width, height)`` to skip file I/O (e.g. when the
    same dimensions are shared across all frames in a scene).
    """
    fx = float(intrinsics.get("fx", 0))
    fy = float(intrinsics.get("fy", 0))
    cx = float(intrinsics.get("cx", 0))
    cy = float(intrinsics.get("cy", 0))
    meta_h = float(intrinsics.get("height", 0))
    meta_w = float(intrinsics.get("width", 0))

    # correct for downsampling if the PNG was saved at a lower resolution
    if meta_h > 0 and meta_w > 0:
        try:
            if img_size is not None:
                actual_w, actual_h = img_size
            elif rgb_path is not None and rgb_path.exists():
                actual_w, actual_h = _png_size(rgb_path)
            else:
                actual_w, actual_h = None, None
            if actual_w and actual_h:
                ds_h = meta_h / actual_h
                ds_w = meta_w / actual_w
                fx /= ds_w;  cx /= ds_w
                fy /= ds_h;  cy /= ds_h
        except Exception:
            pass

    K = [
        [fx,  0.0, cx,  0.0],
        [0.0, fy,  cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return K


def build_cam_tform4x4_obj(cam_meta: dict) -> list:
    """world→camera transform from camera quaternion/translation."""
    q = cam_meta.get("quaternion", [1, 0, 0, 0])   # wxyz
    t = cam_meta.get("translation", [0, 0, 0])
    world_tform4x4_cam = _tform4x4_from_quat_wxyz_and_transl(q, t)
    cam_tform4x4_world = _inv_tform4x4(world_tform4x4_cam)
    return cam_tform4x4_world.tolist()


def build_obj_cam_tform(obj: dict) -> list:
    """camera→object transform from per-object quaternion_wxyz/translation fields."""
    q = obj.get("quaternion_wxyz", [1, 0, 0, 0])
    t = obj.get("translation", [0, 0, 0])
    # Omni6DPose stores cam_tform4x4_obj directly as (q, t) in camera space
    cam_tform4x4_obj = _tform4x4_from_quat_wxyz_and_transl(q, t)

    # apply isotropic scale (mesh units → metres) to the rotation block
    scale = obj.get("meta", {}).get("scale", None)
    if scale is not None:
        s = float(scale[0]) if isinstance(scale, (list, tuple)) else float(scale)
        cam_tform4x4_obj[:3, :3] *= s

    return cam_tform4x4_obj.tolist()


# ── private math helpers ─────────────────────────────────────────────────────

def _tform4x4_from_quat_wxyz_and_transl(q, t) -> torch.Tensor:
    """Build a 4×4 SE(3) matrix from a wxyz quaternion and a 3-vector translation."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

    n = w*w + x*x + y*y + z*z
    if n < 1e-10:
        R = torch.eye(3)
    else:
        s = 2.0 / n
        R = torch.tensor([
            [1 - s*(y*y + z*z),     s*(x*y - w*z),     s*(x*z + w*y)],
            [    s*(x*y + w*z), 1 - s*(x*x + z*z),     s*(y*z - w*x)],
            [    s*(x*z - w*y),     s*(y*z + w*x), 1 - s*(x*x + y*y)],
        ], dtype=torch.float64)

    M = torch.eye(4, dtype=torch.float64)
    M[:3, :3] = R
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M.float()


def _inv_tform4x4(T: torch.Tensor) -> torch.Tensor:
    """Invert an SE(3) 4×4 matrix."""
    R  = T[:3, :3]
    tr = T[:3,  3]
    Rt = R.T
    M  = torch.eye(4, dtype=T.dtype)
    M[:3, :3] = Rt
    M[:3,  3] = -(Rt @ tr)
    return M


def _png_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of a PNG by reading only the 24-byte file header.

    Much faster than loading the full image, especially over NFS.
    PNG spec: bytes 16-19 = width, bytes 20-23 = height (big-endian uint32).
    """
    import struct
    with open(path, "rb") as f:
        f.seek(16)
        data = f.read(8)
    w, h = struct.unpack(">II", data)
    return w, h

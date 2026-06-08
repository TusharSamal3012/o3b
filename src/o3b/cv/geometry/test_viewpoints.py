"""
Tests for get_front_viewpoint / get_front_top_viewpoints / get_front_top_right_viewpoints.

OpenGL convention throughout: +X right, +Y top, +Z back (-Z forward).

Run with:
    pytest src/o3b/cv/geometry/test_viewpoints.py -v
"""
import math
import torch
import pytest

from o3b.data.viz import (
    get_front_viewpoint,
    get_front_top_viewpoints,
    get_front_top_right_viewpoints,
)

DIST = 5.0
TOL  = 1e-5


# ── helpers ───────────────────────────────────────────────────────────────────

def _R(tform):
    return tform[..., :3, :3]

def _t(tform):
    return tform[..., :3, 3]

def _cam_pos(tform):
    """Camera centre in object space = -R.T @ t."""
    return (-_R(tform).T @ _t(tform).unsqueeze(-1)).squeeze(-1)

def _transform_point(tform, pt):
    """Apply 4×4 tform to a 3-vector."""
    ph = torch.cat([pt, torch.ones(1, dtype=pt.dtype)])
    return (tform @ ph)[:3]

def _transform_dir(tform, d):
    """Apply only the rotation part of tform to a direction vector."""
    return _R(tform) @ d

def _assert_valid_rotation(tform, tag=""):
    R = _R(tform)
    I = torch.eye(3, dtype=R.dtype)
    assert torch.allclose(R @ R.T, I, atol=TOL), f"{tag} R @ R.T != I"
    # Proper rotation: right-handed camera frame ⟹ det = +1
    assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0, dtype=R.dtype), atol=TOL), \
        f"{tag} det(R)={torch.linalg.det(R):.6f} != +1"

def _assert_origin_in_front(tform, tag=""):
    z_cam = _transform_point(tform, torch.zeros(3, dtype=tform.dtype))[2]
    assert z_cam < 0, f"{tag} origin z_cam={z_cam:.4f} should be < 0 (in front of camera)"


# ── batch shape ───────────────────────────────────────────────────────────────

def test_batch_shapes():
    assert get_front_viewpoint(dist=DIST).cam_tform4x4_obj.shape          == (1, 4, 4)
    assert get_front_top_viewpoints(dist=DIST).cam_tform4x4_obj.shape     == (2, 4, 4)
    assert get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj.shape == (3, 4, 4)


# ── front view (azim=0, elev=0) ───────────────────────────────────────────────
#
#   pos = (0, 0, -DIST)
#   R   = [[-1,0,0],[0,1,0],[0,0,-1]]   right-handed: x_w×y_w=z_w ✓
#         camera right (+X_cam) = world -X  (object's right mirrors to screen-left)
#         camera up   (+Y_cam) = world +Y
#   t   = (0, 0, -DIST)

def test_front_rotation():
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    _assert_valid_rotation(tform, "front")

    expected_R = torch.tensor([[-1., 0.,  0.],
                                [ 0., 1.,  0.],
                                [ 0., 0., -1.]])
    assert torch.allclose(_R(tform), expected_R, atol=TOL), \
        f"front R mismatch:\n{_R(tform)}"

def test_front_translation():
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    expected_t = torch.tensor([0., 0., -DIST])
    assert torch.allclose(_t(tform), expected_t, atol=TOL), \
        f"front t={_t(tform)}, expected {expected_t}"

def test_front_camera_position():
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    expected_pos = torch.tensor([0., 0., -DIST])
    assert torch.allclose(_cam_pos(tform), expected_pos, atol=TOL)

def test_front_origin_in_front():
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    _assert_origin_in_front(tform, "front")

def test_front_world_x_maps_to_cam_left():
    # Camera faces +Z (looking at object head-on): object's +X mirrors to screen-left (-X_cam)
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    d = _transform_dir(tform, torch.tensor([1., 0., 0.]))
    assert torch.allclose(d, torch.tensor([-1., 0., 0.]), atol=TOL), \
        f"front: world +X → cam {d}, expected (-1,0,0)"

def test_front_world_y_is_cam_up():
    tform = get_front_viewpoint(dist=DIST).cam_tform4x4_obj[0]
    d = _transform_dir(tform, torch.tensor([0., 1., 0.]))
    assert torch.allclose(d, torch.tensor([0., 1., 0.]), atol=TOL), \
        f"front: world +Y → cam {d}, expected (0,1,0)"


# ── top view (azim=0, elev=π/2−0.01) ─────────────────────────────────────────
#
#   pos ≈ (0, DIST, 0)   (slightly tilted; elev is not exactly π/2)

def test_top_rotation_valid():
    tform = get_front_top_viewpoints(dist=DIST).cam_tform4x4_obj[1]
    _assert_valid_rotation(tform, "top")

def test_top_origin_in_front():
    tform = get_front_top_viewpoints(dist=DIST).cam_tform4x4_obj[1]
    _assert_origin_in_front(tform, "top")

def test_top_camera_position():
    tform = get_front_top_viewpoints(dist=DIST).cam_tform4x4_obj[1]
    pos = _cam_pos(tform)
    # Camera is near the top (+Y), with a tiny −Z offset from elev = π/2 − 0.01
    assert abs(pos[0]) < 0.01, f"top cam x={pos[0]:.4f} should ≈ 0"
    assert pos[1] > DIST * 0.99, f"top cam y={pos[1]:.4f} should be ≈ {DIST}"
    assert pos[2] > -DIST * 0.02, f"top cam z={pos[2]:.4f} should be small negative"

def test_top_world_x_maps_to_cam_left():
    # Top-down camera also mirrors X (same right-handed convention): world +X → screen-left
    tform = get_front_top_viewpoints(dist=DIST).cam_tform4x4_obj[1]
    d = _transform_dir(tform, torch.tensor([1., 0., 0.]))
    assert d[0] < -0.99, f"top: world +X x_cam={d[0]:.4f} should be ≈ -1"


# ── right view (azim=π/2, elev=0) ────────────────────────────────────────────
#
#   pos = (-DIST, 0, 0)
#   R   = [[0,0,1],[0,1,0],[-1,0,0]]   right-handed: x_w×y_w=z_w ✓
#         camera right (+X_cam) = world +Z
#         camera up   (+Y_cam) = world +Y
#   t   = (0, 0, -DIST)

def test_right_rotation():
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    _assert_valid_rotation(tform, "right")

    expected_R = torch.tensor([[ 0., 0.,  1.],
                                [ 0., 1.,  0.],
                                [-1., 0.,  0.]])
    # azim=π/2 in float32 leaves a tiny residual (~4e-8) on the sin/cos entries
    assert torch.allclose(_R(tform), expected_R, atol=1e-4), \
        f"right R mismatch:\n{_R(tform)}"

def test_right_translation():
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    expected_t = torch.tensor([0., 0., -DIST])
    assert torch.allclose(_t(tform), expected_t, atol=TOL), \
        f"right t={_t(tform)}, expected {expected_t}"

def test_right_camera_position():
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    expected_pos = torch.tensor([-DIST, 0., 0.])
    assert torch.allclose(_cam_pos(tform), expected_pos, atol=TOL)

def test_right_origin_in_front():
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    _assert_origin_in_front(tform, "right")

def test_right_world_y_is_cam_up():
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    d = _transform_dir(tform, torch.tensor([0., 1., 0.]))
    assert torch.allclose(d, torch.tensor([0., 1., 0.]), atol=TOL), \
        f"right: world +Y → cam {d}, expected (0,1,0)"

def test_right_world_x_points_toward_camera():
    # From the right-side camera, world +X is the depth axis (pointing away from camera,
    # since camera is at x=-DIST). A world point at +X is further from camera → more negative z_cam.
    tform = get_front_top_right_viewpoints(dist=DIST).cam_tform4x4_obj[2]
    z_origin = _transform_point(tform, torch.zeros(3))[2]
    z_xplus  = _transform_point(tform, torch.tensor([1., 0., 0.]))[2]
    assert z_xplus < z_origin, \
        f"right: world +X should increase depth (more negative z_cam), got z_origin={z_origin:.3f}, z_xplus={z_xplus:.3f}"

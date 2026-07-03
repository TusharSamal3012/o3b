from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from o3b.task.task import OD3D_Task, register_task
from o3b.data.datatypes.frame_object import FrameObjectPairBatch
from o3b.task.datatypes.frame_object_pair_quant import FrameObjectPairQuantBatch
from o3b.task.datatypes.frame_object_pair_qualit import FrameObjectPairQualitBatch


# ── projection helpers ─────────────────────────────────────────────────────────

def _proj_frame(kpts3d: torch.Tensor, tform4x4: torch.Tensor, intr4x4: torch.Tensor):
    """Project 3D points to 2D pixels via the frame (CV) convention — matches the
    frame-object overlay viewer. Returns an (K, 2) numpy array."""
    from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
    uv = proj3d2d_tform4x4_intr4x4_broadcast(
        pts3d=kpts3d.float().cpu().unsqueeze(0),
        tform4x4=tform4x4.float().cpu().unsqueeze(0).unsqueeze(0),
        intr4x4=intr4x4.float().cpu().unsqueeze(0).unsqueeze(0),
    )[0]
    return uv.numpy()


def _proj_opengl(kpts3d: torch.Tensor, cam_tform4x4_obj: torch.Tensor, intr4x4: torch.Tensor):
    """Project via the OpenGL convention used by render_trimesh_to_tensor
    (-Z forward, -Y up). Returns an (K, 2) numpy array."""
    import numpy as np
    fx, fy = intr4x4[0, 0].item(), intr4x4[1, 1].item()
    cx, cy = intr4x4[0, 2].item(), intr4x4[1, 2].item()
    pts_h = torch.cat([kpts3d.float().cpu(), torch.ones(len(kpts3d), 1)], dim=1)
    pts_cam = (cam_tform4x4_obj.float().cpu() @ pts_h.T).T[:, :3]
    z = (-pts_cam[:, 2]).clamp(min=1e-6).numpy()
    u = fx * pts_cam[:, 0].numpy() / z + cx
    v = fy * (-pts_cam[:, 1]).numpy() / z + cy
    return np.stack([u, v], axis=1)


def _to_uint8_img(rgb: torch.Tensor):
    """(3, H, W) tensor → (H, W, 3) uint8 numpy."""
    import numpy as np
    a = rgb.float().cpu()
    if a.max() > 1.5:
        a = a / 255.0
    return (a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── qualitative renderers ──────────────────────────────────────────────────────

def _draw_corr_panels(left, right, left_uv, right_gt_uv, right_pred_uv, valid, correct,
                      left_amodal=None, right_amodal=None):
    """Draw a query|target side-by-side panel (left/right are (H, W, 3) uint8):
    each query keypoint (left) linked to its predicted target location (right),
    GT target keypoints shown as hollow white rings.  Returns (3, H, Wl+Wr)."""
    import numpy as np
    from PIL import Image, ImageDraw

    H = left.shape[0]
    if right.shape[0] != H:  # match heights for side-by-side concat
        new_w = max(1, int(round(right.shape[1] * H / right.shape[0])))
        right = np.asarray(Image.fromarray(right).resize((new_w, H)))
    Wl, Wr = left.shape[1], right.shape[1]
    pil = Image.fromarray(np.concatenate([left, right], axis=1))
    draw = ImageDraw.Draw(pil)
    draw.line([(Wl, 0), (Wl, H - 1)], fill=(120, 120, 120), width=2)
    from o3b.cv.visual.draw import get_colors
    K = len(valid)
    kpt_colors_f = get_colors(K)  # (K, 3) float [0, 1], distinct per keypoint
    r = max(3, H // 64)
    for k in range(len(valid)):
        if not bool(valid[k]):
            continue
        col     = (0, 180, 0) if bool(correct[k]) else (0, 100, 220)     # line: green / blue
        kpt_col = tuple((kpt_colors_f[k] * 255).round().int().tolist())   # keypoint: per-index color
        sx, sy = int(np.clip(left_uv[k, 0], 0, Wl - 1)),       int(np.clip(left_uv[k, 1], 0, H - 1))
        gx, gy = int(np.clip(right_gt_uv[k, 0], 0, Wr - 1)),   int(np.clip(right_gt_uv[k, 1], 0, H - 1))
        px, py = int(np.clip(right_pred_uv[k, 0], 0, Wr - 1)), int(np.clip(right_pred_uv[k, 1], 0, H - 1))
        gx += Wl; px += Wl
        rb = max(1, r // 2)  # black centre dot marks an amodal (occluded) keypoint
        draw.line([(sx, sy), (px, py)], fill=col, width=1)                         # query → predicted
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=kpt_col, outline=(0, 0, 0))       # source
        if left_amodal is not None and bool(left_amodal[k]):
            draw.ellipse([sx - rb, sy - rb, sx + rb, sy + rb], fill=(0, 0, 0))
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=kpt_col, width=2)             # GT ring
        if right_amodal is not None and bool(right_amodal[k]):
            draw.ellipse([gx - rb, gy - rb, gx + rb, gy + rb], fill=(0, 0, 0))
        draw.ellipse([px - r, py - r, px + r, py + r], fill=kpt_col, outline=(0, 0, 0))      # predicted
        if right_amodal is not None and bool(right_amodal[k]):
            draw.ellipse([px - rb, py - rb, px + rb, py + rb], fill=(0, 0, 0))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _render_mesh_topdown(mesh, cam_t, cam_k, H, W):
    """Render one mesh from the given (top-down) camera → (H, W, 3) uint8."""
    import numpy as np
    if mesh is None:
        return np.full((H, W, 3), 235, dtype=np.uint8)
    try:
        from o3b.io import _mesh_to_trimesh
        from o3b.cv.visual.show import render_trimesh_to_tensor
        rgb, _ = render_trimesh_to_tensor(
            _mesh_to_trimesh(mesh), cam_k, cam_t, H, W, rgb_bg=[0.92, 0.92, 0.92],
        )
        return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        return np.full((H, W, 3), 235, dtype=np.uint8)


def _topdown_camera(H=256, W=256):
    """A single top-down camera (looking down the NCDS vertical axis) + intrinsics."""
    from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size
    from o3b.cv.geometry.transform import transf4x4_from_spherical
    cam_t = transf4x4_from_spherical(
        azim=torch.tensor([0.0]), elev=torch.tensor([math.pi / 2 - 0.15]),
        theta=torch.tensor([0.0]), dist=5.0,
    )[0].float()
    cam_k = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()
    return cam_t, cam_k


def _render_topdown_corr_img(
    src_mesh, trgt_mesh, src_kpts_ncds, trgt_gt_kpts_ncds, trgt_pred_kpts_ncds,
    valid, correct, H=256, W=256, src_amodal=None, trgt_amodal=None,
):
    """source|target objects rendered from a top-down camera, with source query
    keypoints linked to their predicted target locations.  Returns (3, H, 2W)
    or None if the renderer is unavailable."""
    try:
        from o3b.io import _mesh_to_trimesh  # noqa: F401  (renderer availability probe)
        from o3b.cv.visual.show import render_trimesh_to_tensor  # noqa: F401
    except ImportError:
        return None
    cam_t, cam_k = _topdown_camera(H, W)
    left  = _render_mesh_topdown(src_mesh,  cam_t, cam_k, H, W)
    right = _render_mesh_topdown(trgt_mesh, cam_t, cam_k, H, W)
    src_uv  = _proj_opengl(src_kpts_ncds,       cam_t, cam_k)
    gt_uv   = _proj_opengl(trgt_gt_kpts_ncds,   cam_t, cam_k)
    pred_uv = _proj_opengl(trgt_pred_kpts_ncds, cam_t, cam_k)
    return _draw_corr_panels(left, right, src_uv, gt_uv, pred_uv, valid, correct,
                             left_amodal=src_amodal, right_amodal=trgt_amodal)


# ── correspondence variants ────────────────────────────────────────────────────

def _img_hw(imgs, b):
    """(H, W) of sample b for a stacked tensor or a per-sample list."""
    im = imgs[b]
    return int(im.shape[-2]), int(im.shape[-1])


def _pred_featmap2d(batch, src_ncds, kpts_valid):
    """Variant b: featmap 2D correspondence.

    Per sample: project src keypoints into the src image (GT pose), sample the
    src featmap at those pixels, find the nearest-neighbour pixel in the trgt
    featmap (search restricted to the trgt object mask ∧ valid depth), and lift
    the matched pixel to 3D camera space via the trgt depth map.

    Returns (pred_kpts_cam_t (B,K,3) w/ NaN where unavailable,
             src_uv (B,K,2), pred_uv (B,K,2)) or (None, None, None).
    """
    import torch.nn.functional as F
    from o3b.cv.geometry.transform import depth2pts3d_grid

    if (batch.src_featmap is None or batch.trgt_featmap is None
            or batch.src_rgb is None or batch.trgt_depth is None
            or batch.src_cam_intr4x4 is None or batch.trgt_cam_intr4x4 is None
            or batch.src_obj_kpts3d is None):
        return None, None, None

    B, K, _ = batch.src_obj_kpts3d.shape
    pred_kpts = torch.full((B, K, 3), float("nan"))
    src_uv_all  = torch.full((B, K, 2), float("nan"))
    pred_uv_all = torch.full((B, K, 2), float("nan"))

    for b in range(B):
        try:
            src_fm  = batch.src_featmap[b].float()   # (F, Hf, Wf)
            trgt_fm = batch.trgt_featmap[b].float()  # (F, Hf, Wf)
            Hf, Wf = src_fm.shape[-2:]
            H_s, W_s = _img_hw(batch.src_rgb, b)
            trgt_depth = batch.trgt_depth[b].float().cpu()  # (H, W)
            H_t, W_t = trgt_depth.shape[-2:]

            # 1. src kpts → 2D (image res, GT pose)
            src_uv = torch.from_numpy(_proj_frame(
                batch.src_obj_kpts3d[b], src_ncds[b], batch.src_cam_intr4x4[b],
            )).float()  # (K, 2)
            src_uv_all[b] = src_uv

            # 2. sample src feats at kpt pixels (featmap res)
            uf = ((src_uv[:, 0] + 0.5) * Wf / W_s - 0.5).round().long().clamp(0, Wf - 1)
            vf = ((src_uv[:, 1] + 0.5) * Hf / H_s - 0.5).round().long().clamp(0, Hf - 1)
            query_feats = src_fm[:, vf, uf].T  # (K, F)

            # 3. search mask on the trgt featmap: object mask ∧ valid depth
            valid_t = (trgt_depth > 0)
            if batch.trgt_fo_mask is not None:
                valid_t = valid_t & batch.trgt_fo_mask[b].bool().cpu()
            mask_f = F.interpolate(
                valid_t[None, None].float(), size=(Hf, Wf), mode="nearest",
            )[0, 0] > 0.5  # (Hf, Wf)
            if not mask_f.any():
                mask_f = torch.ones_like(mask_f)

            # 4. NN in feature space (feats are L2-normalised → cosine sim)
            sim = torch.einsum("kf,fhw->khw", query_feats, trgt_fm)  # (K, Hf, Wf)
            sim = sim.masked_fill(~mask_f[None], -float("inf"))
            flat_idx = sim.reshape(K, -1).argmax(dim=1)  # (K,)
            pv, pu = flat_idx // Wf, flat_idx % Wf

            # 5. featmap pixel → image res
            pu_img = (pu.float() + 0.5) * W_t / Wf - 0.5
            pv_img = (pv.float() + 0.5) * H_t / Hf - 0.5
            pred_uv_all[b] = torch.stack([pu_img, pv_img], dim=1)

            # 6. lift via trgt depth (OpenGL cam space, matching the GT kpts)
            pts3d = depth2pts3d_grid(
                trgt_depth[None], batch.trgt_cam_intr4x4[b].float().cpu(), opengl=True,
            )  # (3, H, W)
            iu = pu_img.round().long().clamp(0, W_t - 1)
            iv = pv_img.round().long().clamp(0, H_t - 1)
            pred_kpts[b] = pts3d[:, iv, iu].T  # (K, 3)
        except Exception:
            continue

    return pred_kpts, src_uv_all, pred_uv_all


def _pred_mesh_corresp(batch, query_kpts_cam_q, pred_src, pred_trgt):
    """Variant c: mesh (barycentric) correspondence.

    Per sample: pose the src/trgt meshes into their camera spaces with the
    predicted poses, project each query keypoint onto the src mesh surface,
    express it in barycentric coordinates of its triangle, and evaluate the same
    triangle + barycentric coordinates on the trgt mesh.

    Requires src and trgt meshes with identical topology (same category mesh).
    Returns pred_kpts_cam_t (B, K, 3) with NaN where unavailable, or None.
    """
    src_meshes  = batch.src_pred_mesh  if batch.src_pred_mesh  is not None else batch.src_meshes
    trgt_meshes = batch.trgt_pred_mesh if batch.trgt_pred_mesh is not None else batch.trgt_meshes
    if src_meshes is None or trgt_meshes is None:
        return None

    import numpy as np
    from o3b.cv.geometry.transform import transf3d_broadcast

    B, K, _ = query_kpts_cam_q.shape
    pred_kpts = torch.full((B, K, 3), float("nan"))

    for b in range(B):
        try:
            m_src, m_trgt = src_meshes[b], trgt_meshes[b]
            if m_src is None or m_trgt is None:
                continue
            f_src  = m_src.faces.long().cpu()
            f_trgt = m_trgt.faces.long().cpu()
            if f_src.shape != f_trgt.shape or len(m_src.verts) != len(m_trgt.verts):
                continue  # barycentric transfer needs identical topology

            v_src_cam  = transf3d_broadcast(m_src.verts.float().cpu(),  pred_src[b].float().cpu())
            v_trgt_cam = transf3d_broadcast(m_trgt.verts.float().cpu(), pred_trgt[b].float().cpu())

            import trimesh
            from trimesh.triangles import points_to_barycentric, barycentric_to_points
            tm = trimesh.Trimesh(v_src_cam.numpy(), f_src.numpy(), process=False)
            closest, _dists, face_ids = tm.nearest.on_surface(
                query_kpts_cam_q[b].detach().cpu().numpy(),
            )
            bary = points_to_barycentric(
                triangles=v_src_cam.numpy()[f_src.numpy()[face_ids]], points=closest,
            )
            pred = barycentric_to_points(
                triangles=v_trgt_cam.numpy()[f_trgt.numpy()[face_ids]], barycentric=bary,
            )
            pred_kpts[b] = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        except Exception:
            continue

    return pred_kpts


def _variant_metrics(pred_kpts_cam_t, gt_kpts_cam_t, kpts_valid, threshold,
                     src_vis, trgt_vis):
    """Euclidean dist / PCK@0.1 / modal / amodal PCK for one prediction variant.

    NaN predictions (variant unavailable for that sample) yield NaN aggregates so
    nan-safe means skip them.  Returns a dict of tensors + per-kpt is_correct.
    """
    dev = gt_kpts_cam_t.device
    pred = pred_kpts_cam_t.to(dev)
    dist = (pred - gt_kpts_cam_t).norm(dim=-1)                       # (B, K)
    sample_ok = torch.isfinite(dist).all(dim=1)                      # (B,)
    n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)

    is_correct = (dist < threshold.unsqueeze(1)) & torch.isfinite(dist)
    correct = is_correct.float() * kpts_valid.float()

    dist_masked = torch.where(kpts_valid, dist, torch.zeros_like(dist))
    dist_mean = torch.where(sample_ok, dist_masked.sum(dim=1) / n_valid,
                            torch.full_like(n_valid, float("nan")))
    pck = torch.where(sample_ok, correct.sum(dim=1) / n_valid,
                      torch.full_like(n_valid, float("nan")))

    modal_pck = amodal_pck = None
    if src_vis is not None and trgt_vis is not None:
        both_vis    = src_vis.bool().to(dev) & trgt_vis.bool().to(dev)
        modal_mask  = kpts_valid & both_vis
        amodal_mask = kpts_valid & ~both_vis
        n_modal, n_amodal = modal_mask.float().sum(dim=1), amodal_mask.float().sum(dim=1)
        nan = torch.full_like(n_valid, float("nan"))
        modal_pck  = torch.where((n_modal  > 0) & sample_ok,
                                 (correct * modal_mask.float()).sum(dim=1)  / n_modal.clamp(min=1),  nan)
        amodal_pck = torch.where((n_amodal > 0) & sample_ok,
                                 (correct * amodal_mask.float()).sum(dim=1) / n_amodal.clamp(min=1), nan)

    return {"dist": dist, "dist_mean": dist_mean, "pck": pck,
            "modal": modal_pck, "amodal": amodal_pck, "is_correct": is_correct}


# ── task ───────────────────────────────────────────────────────────────────────

@register_task("CamCrsp3DNNTask")
class CamCrsp3DNNTask(OD3D_Task):
    """Camera-space 3D correspondence driven by predicted object poses.

    For a frame-object pair (query = src, target = trgt) the GT keypoints live
    in NCDS object space.  The metric goes:

      1. Transform query and target ``obj_kpts3d`` into their respective camera
         spaces via the GT ``cam_tform4x4_obj_ncds``.
      2. From the *predicted* per-frame poses build the relative camera
         transform::

             cam_query_tform4x4_cam_target = pred_cam_q_tform_obj @ inv(pred_cam_t_tform_obj)
             cam_target_tform4x4_cam_query = inv(cam_query_tform4x4_cam_target)

         (the relative cam↔cam transform is rigid, so any uniform object scale
         embedded in the poses cancels).
      3. Map the query keypoints (query-camera space) into the target camera
         space and compare against the GT target keypoints.
      4. Euclidean error, thresholded at ``0.1 * trgt_obj_size`` for PCK.

    Qualitative: correspondences drawn on the query|target frame RGB, and on a
    top-down render of the target object.  Predicted poses fall back to GT when
    unavailable (oracle upper bound).
    """

    def __init__(self, qualit: bool = True, **kwargs):
        self.qualit = qualit

    def forward(self, batch: FrameObjectPairBatch, return_qualit: bool = True) -> Tuple[FrameObjectPairQuantBatch, FrameObjectPairQualitBatch]:
        from o3b.cv.geometry.transform import (
            transf3d_broadcast, tform4x4_broadcast, inv_tform4x4,
        )

        src_kpts       = batch.src_obj_kpts3d
        trgt_kpts      = batch.trgt_obj_kpts3d
        src_kpts_mask  = batch.src_obj_kpts3d_mask
        trgt_kpts_mask = batch.trgt_obj_kpts3d_mask
        src_ncds       = batch.src_cam_tform4x4_obj_ncds
        trgt_ncds      = batch.trgt_cam_tform4x4_obj_ncds
        trgt_obj_size  = batch.trgt_obj_size

        # Predicted poses, falling back to GT (oracle) when unavailable. The oracle
        # uses cam_tform4x4_obj_ncds (NCDS→cam), which embeds the 1D size (largest
        # object dimension) so cross-instance correspondences transfer in a
        # size-normalised frame rather than in absolute metric pose.
        pred_src  = batch.src_pred_cam_tform4x4_obj  if batch.src_pred_cam_tform4x4_obj  is not None else src_ncds
        pred_trgt = batch.trgt_pred_cam_tform4x4_obj if batch.trgt_pred_cam_tform4x4_obj is not None else trgt_ncds

        required = (src_kpts, trgt_kpts, src_ncds, trgt_ncds, pred_src, pred_trgt, trgt_obj_size)
        if any(x is None for x in required):
            return FrameObjectPairQuantBatch(), FrameObjectPairQualitBatch()

        B, K, _ = src_kpts.shape
        dev = src_kpts.device

        # ── valid keypoint mask ───────────────────────────────────────────────
        if src_kpts_mask is not None and trgt_kpts_mask is not None:
            kpts_valid = src_kpts_mask & trgt_kpts_mask
        elif src_kpts_mask is not None:
            kpts_valid = src_kpts_mask
        elif trgt_kpts_mask is not None:
            kpts_valid = trgt_kpts_mask
        else:
            kpts_valid = torch.ones(B, K, dtype=torch.bool, device=dev)
        n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)

        # ── step 1: kpts (NCDS) → camera space (metric, GT poses) ─────────────
        query_kpts_cam_q   = transf3d_broadcast(src_kpts.float(),  src_ncds.float().unsqueeze(1))   # (B, K, 3)
        gt_trgt_kpts_cam_t = transf3d_broadcast(trgt_kpts.float(), trgt_ncds.float().unsqueeze(1))   # (B, K, 3)

        # ── step 2: predicted relative camera transform (target ← query) ──────
        cam_query_tform4x4_cam_target = tform4x4_broadcast(pred_src.float(), inv_tform4x4(pred_trgt.float()))
        cam_target_tform4x4_cam_query = inv_tform4x4(cam_query_tform4x4_cam_target)  # (B, 4, 4)

        # ── step 3: query kpts → predicted target camera space ────────────────
        pred_trgt_kpts_cam_t = transf3d_broadcast(
            query_kpts_cam_q, cam_target_tform4x4_cam_query.unsqueeze(1),
        )  # (B, K, 3)

        # ── variant b: featmap 2D correspondence (NN in feature space + depth) ─
        pred_feat2d_kpts, feat2d_src_uv, feat2d_pred_uv = _pred_featmap2d(
            batch, src_ncds, kpts_valid,
        )

        # ── variant c: mesh (barycentric) correspondence ───────────────────────
        pred_mesh_kpts = _pred_mesh_corresp(batch, query_kpts_cam_q, pred_src, pred_trgt)

        # ── metrics: euclidean error + PCK @ 0.1 * trgt_obj_size, per variant ──
        threshold = 0.1 * trgt_obj_size.float().to(dev)  # (B,)
        src_vis, trgt_vis = batch.src_obj_kpts2d_mask, batch.trgt_obj_kpts2d_mask

        m_pose = _variant_metrics(pred_trgt_kpts_cam_t, gt_trgt_kpts_cam_t,
                                  kpts_valid, threshold, src_vis, trgt_vis)

        quant = FrameObjectPairQuantBatch(
            cam_kpts_trgt_euc_dist      = m_pose["dist"],
            kpts_mask                   = kpts_valid,
            cam_kpts_trgt_euc_dist_mean = m_pose["dist_mean"],
            cam_kpts_trgt_pck01         = m_pose["pck"],
            cam_kpts_modal_trgt_pck01   = m_pose["modal"],
            cam_kpts_amodal_trgt_pck01  = m_pose["amodal"],
        )

        variant_correct = {"pose": m_pose["is_correct"]}
        for name, pred in (("featmap2d", pred_feat2d_kpts), ("mesh", pred_mesh_kpts)):
            if pred is None:
                continue
            m = _variant_metrics(pred, gt_trgt_kpts_cam_t, kpts_valid, threshold,
                                 src_vis, trgt_vis)
            variant_correct[name] = m["is_correct"]
            quant.extra[f"cam_kpts_trgt_euc_dist_mean_{name}"] = m["dist_mean"]
            quant.extra[f"cam_kpts_trgt_pck01_{name}"]         = m["pck"]
            if m["modal"] is not None:
                quant.extra[f"cam_kpts_modal_trgt_pck01_{name}"]  = m["modal"]
                quant.extra[f"cam_kpts_amodal_trgt_pck01_{name}"] = m["amodal"]

        if self.qualit and return_qualit:
            qualit = self._render_qualit(
                batch, query_kpts_cam_q, gt_trgt_kpts_cam_t, pred_trgt_kpts_cam_t,
                trgt_ncds, kpts_valid, m_pose["is_correct"],
            )
            self._render_qualit_featmap2d(
                batch, qualit, src_ncds, feat2d_src_uv, feat2d_pred_uv,
                kpts_valid, variant_correct.get("featmap2d"),
            )
            self._render_qualit_mesh(
                batch, qualit, src_ncds, trgt_ncds, pred_src, pred_trgt,
                pred_mesh_kpts, kpts_valid, variant_correct.get("mesh"),
            )
        else:
            qualit = None

        return quant, qualit

    # ── qualitative ────────────────────────────────────────────────────────────

    def _render_qualit(
        self, batch, query_kpts_cam_q, gt_trgt_kpts_cam_t, pred_trgt_kpts_cam_t,
        trgt_ncds, kpts_valid, is_correct,
    ) -> FrameObjectPairQualitBatch:
        import torch.nn.functional as F
        from o3b.cv.geometry.transform import transf3d_broadcast, inv_tform4x4

        B = query_kpts_cam_q.shape[0]
        eye = torch.eye(4)

        # predicted target keypoints back in target NCDS space (for the render)
        trgt_ncds_inv = inv_tform4x4(trgt_ncds.float())
        pred_kpts_ncds = transf3d_broadcast(pred_trgt_kpts_cam_t, trgt_ncds_inv.unsqueeze(1))  # (B, K, 3)

        # draw the predicted 3-D bounding box on each frame (camera transforms +
        # predicted obj sizes in 3D), via batch_draw_bbox3d (OpenGL convention by
        # default), before the correspondence overlay. The box is expressed in
        # NCDS units (longest side → obj_size_ncds) and posed with the predicted
        # cam_tform4x4_obj_ncds (the GT NCDS pose/size is used as the oracle
        # fallback when no prediction is available).
        _OBJ_SIZE_NCDS = 2.0

        def _frames_with_bbox(rgb, bbox3d, intr, pose_ncds):
            frames = list(rgb) if rgb is not None else None
            if frames is None or bbox3d is None or intr is None:
                return frames
            bbox3d_f = bbox3d.float()
            if bbox3d_f.dim() == 3 and bbox3d_f.shape[1:] == (8, 3):
                # (B, 8, 3) cam-space corners — project and draw directly
                try:
                    from o3b.cv.visual.draw import draw_bbox3d_corners
                    for b_i in range(len(frames)):
                        frames[b_i] = draw_bbox3d_corners(
                            frames[b_i], bbox3d_f[b_i], intr[b_i], thickness=2,
                        ).float().div(255.0)
                except Exception:
                    pass
            elif pose_ncds is not None:
                # (B, 3) NCDS side lengths — normalize and use batch_draw_bbox3d
                try:
                    from o3b.cv.visual.draw import batch_draw_bbox3d
                    bbox3d_r = bbox3d_f.reshape(-1, 3)
                    denom = bbox3d_r.max(dim=1, keepdim=True).values.clamp(min=1e-6)
                    bbox3d_ncds = bbox3d_r * (_OBJ_SIZE_NCDS / denom)
                    frames = batch_draw_bbox3d(frames, bbox3d_ncds, intr, pose_ncds.float(), thickness=2)
                except Exception:
                    frames = list(rgb)
            return frames

        def _pick(pred, gt):
            return pred if pred is not None else gt

        src_frames  = _frames_with_bbox(
            batch.src_rgb,
            _pick(batch.src_pred_obj_size3d, batch.src_cam_bbox3d),
            batch.src_cam_intr4x4,
            _pick(batch.src_pred_cam_tform4x4_obj, batch.src_cam_tform4x4_obj_ncds))
        trgt_frames = _frames_with_bbox(
            batch.trgt_rgb,
            _pick(batch.trgt_pred_obj_size3d, batch.trgt_cam_bbox3d),
            batch.trgt_cam_intr4x4,
            _pick(batch.trgt_pred_cam_tform4x4_obj, batch.trgt_cam_tform4x4_obj_ncds))

        def _vstack(rows):
            """Pad row widths to the max and stack vertically → (3, sum H, W)."""
            rows = [r for r in rows if r is not None]
            if not rows:
                return None
            W = max(r.shape[2] for r in rows)
            out = []
            for r in rows:
                if r.shape[2] != W:
                    new_h = max(1, int(round(r.shape[1] * W / r.shape[2])))
                    r = F.interpolate(r.unsqueeze(0), size=(new_h, W), mode="bilinear", align_corners=False)[0]
                out.append(r)
            return torch.cat(out, dim=1)

        # amodal = annotated (valid) but NOT visible in the frame (obj_kpts2d_mask=False)
        def _amodal(valid_b, vis2d, b):
            if vis2d is None:
                return None
            return valid_b & (~vis2d[b].bool().cpu())

        imgs = []
        for b in range(B):
            valid   = kpts_valid[b].cpu()
            correct = is_correct[b].cpu()
            src_amodal  = _amodal(valid, batch.src_obj_kpts2d_mask, b)
            trgt_amodal = _amodal(valid, batch.trgt_obj_kpts2d_mask, b)

            # ── top row: correspondences on the frame RGB ─────────────────────
            frame_row = None
            if (batch.src_rgb is not None and batch.trgt_rgb is not None
                    and batch.src_cam_intr4x4 is not None and batch.trgt_cam_intr4x4 is not None):
                try:
                    src_uv  = _proj_frame(batch.src_obj_kpts3d[b], batch.src_cam_tform4x4_obj_ncds[b], batch.src_cam_intr4x4[b])
                    gt_uv   = _proj_frame(batch.trgt_obj_kpts3d[b], batch.trgt_cam_tform4x4_obj_ncds[b], batch.trgt_cam_intr4x4[b])
                    pred_uv = _proj_frame(pred_trgt_kpts_cam_t[b], eye, batch.trgt_cam_intr4x4[b])
                    frame_row = _draw_corr_panels(
                        _to_uint8_img(src_frames[b]), _to_uint8_img(trgt_frames[b]),
                        src_uv, gt_uv, pred_uv, valid, correct,
                        left_amodal=src_amodal, right_amodal=trgt_amodal,
                    )
                except Exception:
                    frame_row = None

            # ── bottom row: top-down renders of source + target objects ───────
            src_mesh  = batch.src_meshes[b]  if batch.src_meshes  is not None else None
            trgt_mesh = batch.trgt_meshes[b] if batch.trgt_meshes is not None else None
            top_row = _render_topdown_corr_img(
                src_mesh, trgt_mesh,
                batch.src_obj_kpts3d[b], batch.trgt_obj_kpts3d[b], pred_kpts_ncds[b],
                valid, correct, src_amodal=src_amodal, trgt_amodal=trgt_amodal,
            )

            combined = _vstack([frame_row, top_row])
            if combined is not None:
                imgs.append(combined)

        if not imgs:
            return FrameObjectPairQualitBatch()
        H = max(i.shape[1] for i in imgs)
        W = max(i.shape[2] for i in imgs)
        imgs = [
            i if (i.shape[1] == H and i.shape[2] == W)
            else F.interpolate(i.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            for i in imgs
        ]
        return FrameObjectPairQualitBatch(imgs=torch.stack(imgs))

    # ── qualitative: variant b — featmap PCA correspondence panels ─────────────

    @staticmethod
    def _featmap_pca_imgs(src_fm, trgt_fm, out_hw_src, out_hw_trgt):
        """Joint PCA over both featmaps → two (H, W, 3) uint8 images."""
        import numpy as np
        import torch.nn.functional as F
        Fd, Hf, Wf = src_fm.shape
        flat = torch.cat([
            src_fm.reshape(Fd, -1).T, trgt_fm.reshape(Fd, -1).T,
        ], dim=0)  # (2*Hf*Wf, F)
        with torch.no_grad():
            _, _, Vt = torch.pca_lowrank(flat, q=3, center=True)
            proj = flat @ Vt  # (2*Hf*Wf, 3)
        lo, hi = proj.min(dim=0).values, proj.max(dim=0).values
        proj = (proj - lo) / (hi - lo + 1e-6)
        src_pca  = proj[: Hf * Wf].T.reshape(3, Hf, Wf)
        trgt_pca = proj[Hf * Wf:].T.reshape(3, Hf, Wf)
        out = []
        for pca, (H, W) in ((src_pca, out_hw_src), (trgt_pca, out_hw_trgt)):
            up = F.interpolate(pca[None], size=(H, W), mode="nearest")[0]
            out.append((up.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        return out

    def _render_qualit_featmap2d(
        self, batch, qualit, src_ncds, feat2d_src_uv, feat2d_pred_uv,
        kpts_valid, is_correct,
    ) -> None:
        """query|target joint-PCA featmap panels with the featmap-NN
        correspondences drawn on top → qualit.extra['correspondences_featmap2d']."""
        if (qualit is None or is_correct is None or feat2d_src_uv is None
                or batch.src_featmap is None or batch.trgt_featmap is None
                or batch.trgt_cam_intr4x4 is None):
            return
        B = kpts_valid.shape[0]
        imgs = []
        for b in range(B):
            try:
                hw_s = _img_hw(batch.src_rgb, b)
                hw_t = _img_hw(batch.trgt_rgb, b) if batch.trgt_rgb is not None else hw_s
                left, right = self._featmap_pca_imgs(
                    batch.src_featmap[b].float().cpu(),
                    batch.trgt_featmap[b].float().cpu(), hw_s, hw_t,
                )
                gt_uv = _proj_frame(
                    batch.trgt_obj_kpts3d[b], batch.trgt_cam_tform4x4_obj_ncds[b],
                    batch.trgt_cam_intr4x4[b],
                )
                row = _draw_corr_panels(
                    left, right,
                    feat2d_src_uv[b].numpy(), gt_uv, feat2d_pred_uv[b].numpy(),
                    kpts_valid[b].cpu(), is_correct[b].cpu(),
                )
                imgs.append(row)
            except Exception:
                continue
        if imgs:
            qualit.extra["correspondences_featmap2d"] = imgs

    # ── qualitative: variant c — NOCS-colored mesh correspondence panels ───────

    def _render_qualit_mesh(
        self, batch, qualit, src_ncds, trgt_ncds, pred_src, pred_trgt,
        pred_mesh_kpts, kpts_valid, is_correct,
    ) -> None:
        """NOCS-colored predicted meshes rendered from the GT viewpoints (row 1)
        and top-down (row 2), with the mesh-barycentric correspondences drawn on
        top → qualit.extra['correspondences_mesh']."""
        if qualit is None or is_correct is None or pred_mesh_kpts is None:
            return
        from dataclasses import replace as _dc_replace
        import torch.nn.functional as F
        from o3b.cv.geometry.transform import transf3d_broadcast, inv_tform4x4
        from o3b.cv.visual.point3d_to_color3d import nocs_0c_to_rgb

        src_meshes  = batch.src_pred_mesh  if batch.src_pred_mesh  is not None else batch.src_meshes
        trgt_meshes = batch.trgt_pred_mesh if batch.trgt_pred_mesh is not None else batch.trgt_meshes
        if src_meshes is None or trgt_meshes is None:
            return

        def _nocs_colored(mesh):
            if mesh is None:
                return None
            verts = mesh.verts.float().cpu()
            return _dc_replace(mesh, vert_colors=nocs_0c_to_rgb(verts).clamp(0, 1))

        def _vstack(rows):
            rows = [r for r in rows if r is not None]
            if not rows:
                return None
            W = max(r.shape[2] for r in rows)
            out = []
            for r in rows:
                if r.shape[2] != W:
                    new_h = max(1, int(round(r.shape[1] * W / r.shape[2])))
                    r = F.interpolate(r.unsqueeze(0), size=(new_h, W), mode="bilinear", align_corners=False)[0]
                out.append(r)
            return torch.cat(out, dim=1)

        eye = torch.eye(4)
        B = kpts_valid.shape[0]
        imgs = []
        for b in range(B):
            try:
                m_src  = _nocs_colored(src_meshes[b])
                m_trgt = _nocs_colored(trgt_meshes[b])
                if m_src is None or m_trgt is None:
                    continue
                valid, correct = kpts_valid[b].cpu(), is_correct[b].cpu()

                # ── row 1: GT viewpoints ───────────────────────────────────────
                gt_row = None
                if batch.src_cam_intr4x4 is not None and batch.trgt_cam_intr4x4 is not None:
                    H_s, W_s = _img_hw(batch.src_rgb, b) if batch.src_rgb is not None else (256, 256)
                    H_t, W_t = _img_hw(batch.trgt_rgb, b) if batch.trgt_rgb is not None else (256, 256)
                    src_k,  trgt_k  = batch.src_cam_intr4x4[b].float().cpu(), batch.trgt_cam_intr4x4[b].float().cpu()
                    src_t,  trgt_t  = src_ncds[b].float().cpu(), trgt_ncds[b].float().cpu()
                    left  = _render_mesh_topdown(m_src,  src_t,  src_k,  H_s, W_s)
                    right = _render_mesh_topdown(m_trgt, trgt_t, trgt_k, H_t, W_t)
                    src_uv  = _proj_opengl(batch.src_obj_kpts3d[b],  src_t,  src_k)
                    gt_uv   = _proj_opengl(batch.trgt_obj_kpts3d[b], trgt_t, trgt_k)
                    pred_uv = _proj_opengl(pred_mesh_kpts[b], eye, trgt_k)  # already trgt-cam space
                    gt_row = _draw_corr_panels(left, right, src_uv, gt_uv, pred_uv, valid, correct)

                # ── row 2: top-down ────────────────────────────────────────────
                pred_kpts_ncds = transf3d_broadcast(
                    pred_mesh_kpts[b].float(), inv_tform4x4(pred_trgt[b].float().cpu()),
                )
                top_row = _render_topdown_corr_img(
                    m_src, m_trgt,
                    batch.src_obj_kpts3d[b], batch.trgt_obj_kpts3d[b], pred_kpts_ncds,
                    valid, correct,
                )

                combined = _vstack([gt_row, top_row])
                if combined is not None:
                    imgs.append(combined)
            except Exception:
                continue
        if imgs:
            qualit.extra["correspondences_mesh"] = imgs

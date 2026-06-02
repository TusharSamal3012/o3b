from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from od3d_basic.task.task import OD3D_Task, register_task
from od3d_basic.data.datatypes.object import ObjectPairBatch
from od3d_basic.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from od3d_basic.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


def _render_corr_imgs(
    src_verts:   Tensor,            # (B, V, 3)
    trgt_verts:  Tensor,            # (B, V, 3)
    src_kpts:    Tensor,            # (B, K, 3)
    trgt_kpts:   Tensor,            # (B, K, 3)
    pred_src:    Tensor,            # (B, K, 3) predicted src positions
    valid:       Tensor,            # (B, K) bool
    src_mask:    Optional[Tensor],  # (B, V) bool — non-padding vertices
    trgt_mask:   Optional[Tensor],  # (B, V) bool — non-padding vertices
    src_meshes:  Optional[list] = None,   # list of B Mesh objects (per-sample)
    trgt_meshes: Optional[list] = None,   # list of B Mesh objects (per-sample)
    H: int = 256,
    W: int = 256,
) -> Optional[Tensor]:
    """Return (B, 3, H, 2*W) float32 [0,1] images.

    Left panel: target mesh with target keypoints (filled circles).
    Right panel: source mesh with GT src kpts (open rings) and
                 predicted src positions (filled circles).
    Lines connect each target keypoint to its predicted source position:
        green = correct (error < 0.1 * trgt_max_dim), red = wrong.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from od3d_basic.io import _mesh_to_trimesh
        from od3d_basic.cv.visual.show import (
            render_trimesh_to_tensor,
            get_default_camera_intrinsics_from_img_size,
        )
        from od3d_basic.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
    except ImportError:
        return None

    B, K, _ = src_kpts.shape
    imgs_out = []
    r_dot = max(4, H // 52)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, H // 24))
    except Exception:
        font = ImageFont.load_default()

    # two cameras: front-right and back-left (objects in normalized [-1,1] space)
    cam_ts = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=2, dist=5.0).float()
    cam_k  = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()
    fx  = cam_k[0, 0].item()
    fy  = cam_k[1, 1].item()
    cx_k = cam_k[0, 2].item()
    cy_k = cam_k[1, 2].item()

    def project(pts_3d: Tensor, cam_t: Tensor):
        """Project (N, 3) points → (u_arr, v_arr) pixel arrays (Open3D cam convention)."""
        pts_h = torch.cat([pts_3d.float().cpu(), torch.ones(len(pts_3d), 1)], dim=1)
        pts_cam = (cam_t @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2].clamp(min=1e-6).numpy()
        u = fx * pts_cam[:, 0].numpy() / z + cx_k
        v = fy * pts_cam[:, 1].numpy() / z + cy_k
        return u, v

    def render_panel(mesh, cam_t: Tensor) -> np.ndarray:
        """Render one panel from cam_t; returns (H, W, 3) uint8."""
        if mesh is not None:
            try:
                rgb, _ = render_trimesh_to_tensor(
                    _mesh_to_trimesh(mesh), cam_k, cam_t, H, W,
                    rgb_bg=[0.92, 0.92, 0.92],
                )
                return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            except Exception:
                pass
        return np.full((H, W, 3), 235, dtype=np.uint8)

    for b in range(B):
        sk = src_kpts[b].float().cpu()
        tk = trgt_kpts[b].float().cpu()
        pk = pred_src[b].float().cpu()
        vm = valid[b].cpu()

        # per-sample trgt bounding-box size for correctness threshold
        tv_b = trgt_verts[b].float().cpu()
        if trgt_mask is not None:
            tv_b = tv_b[trgt_mask[b].cpu()]
        trgt_max_dim = (
            (tv_b.max(0).values - tv_b.min(0).values).max().item()
            if tv_b.shape[0] > 0 else 1.0
        )
        threshold = 0.1 * trgt_max_dim
        is_correct = ((pk - sk).norm(dim=-1) < threshold)  # (K,)

        sm = src_meshes[b]  if (src_meshes  is not None) else None
        tm = trgt_meshes[b] if (trgt_meshes is not None) else None

        # build one row per viewpoint, stack vertically → (2H, 2W, 3)
        rows = []
        for v_idx in range(cam_ts.shape[0]):
            cam_t = cam_ts[v_idx]

            trgt_panel = render_panel(tm, cam_t)
            src_panel  = render_panel(sm, cam_t)

            row_arr = np.concatenate([trgt_panel, src_panel], axis=1)  # (H, 2W, 3)
            row_pil = Image.fromarray(row_arr)
            draw = ImageDraw.Draw(row_pil)

            draw.line([(W, 0), (W, H - 1)], fill=(120, 120, 120), width=2)

            tk_u, tk_v = project(tk, cam_t)
            pk_u, pk_v = project(pk, cam_t)
            sk_u, sk_v = project(sk, cam_t)

            for k_i in range(K):
                if not bool(vm[k_i]):
                    continue
                color = (0, 180, 0) if bool(is_correct[k_i]) else (0, 0, 200)

                tku = int(np.clip(tk_u[k_i], 0, W - 1))
                tkv = int(np.clip(tk_v[k_i], 0, H - 1))
                pku = int(np.clip(pk_u[k_i], 0, W - 1)) + W
                pkv = int(np.clip(pk_v[k_i], 0, H - 1))
                sku = int(np.clip(sk_u[k_i], 0, W - 1)) + W
                skv = int(np.clip(sk_v[k_i], 0, H - 1))

                label = str(k_i)
                draw.line([(tku, tkv), (pku, pkv)], fill=color, width=1)
                draw.ellipse(
                    [(tku - r_dot, tkv - r_dot), (tku + r_dot, tkv + r_dot)],
                    fill=color, outline=(0, 0, 0),
                )
                draw.text((tku + r_dot + 2, tkv - r_dot), label, fill=color, font=font)
                draw.ellipse(
                    [(pku - r_dot, pkv - r_dot), (pku + r_dot, pkv + r_dot)],
                    fill=color, outline=(0, 0, 0),
                )
                draw.text((pku + r_dot + 2, pkv - r_dot), label, fill=color, font=font)
                draw.ellipse(
                    [(sku - r_dot, skv - r_dot), (sku + r_dot, skv + r_dot)],
                    outline=color, width=2,
                )
                draw.text((sku + r_dot + 2, skv - r_dot), label, fill=color, font=font)

            rows.append(np.array(row_pil))

        canvas = np.concatenate(rows, axis=0)  # (2H, 2W, 3)
        arr = canvas.astype(np.float32) / 255.0
        imgs_out.append(torch.from_numpy(arr).permute(2, 0, 1))

    return torch.stack(imgs_out)  # (B, 3, 2H, 2W)


@register_task("Crsp3DNNTask")
class Crsp3DNNTask(OD3D_Task):
    """Feature-based nearest-neighbour 3-D correspondence baseline.

    For each target keypoint:
      1. find nearest target vertex by 3-D Euclidean distance
      2. retrieve that vertex's feature vector
      3. find the nearest source vertex in feature space
      4. use that source vertex's 3-D position as the predicted correspondence
    Evaluates euclidean error vs. ground-truth semantic keypoint positions and
    reports PCK01 (fraction of predictions within 0.1 * largest target dimension).
    """

    def __init__(self, **kwargs):
        pass

    def forward(
        self,
        batch: ObjectPairBatch,
    ) -> Tuple[ObjectPairQuantBatch, ObjectPairQualitBatch]:

        src_kpts         = batch.src_obj_kpts3d         # (B, K, 3)
        trgt_kpts        = batch.trgt_obj_kpts3d        # (B, K, 3)
        src_mask         = batch.src_obj_kpts3d_mask    # (B, K) bool or None
        trgt_mask        = batch.trgt_obj_kpts3d_mask   # (B, K) bool or None
        src_verts        = batch.src_verts3d            # (B, V, 3)
        src_feats        = batch.src_verts3d_feats      # (B, V, F)
        trgt_verts       = batch.trgt_verts3d           # (B, V, 3)
        trgt_feats       = batch.trgt_verts3d_feats     # (B, V, F)

        if any(x is None for x in (src_kpts, trgt_kpts, src_verts, src_feats, trgt_verts, trgt_feats)):
            return ObjectPairQuantBatch(), ObjectPairQualitBatch()

        B, K, _ = src_kpts.shape
        dev = src_kpts.device

        trgt_verts_mask = batch.trgt_verts3d_feats_mask  # (B, V) bool or None
        src_verts_mask  = batch.src_verts3d_feats_mask   # (B, V) bool or None

        # ── 1. trgt keypoint → nearest trgt vertex (3D) ───────────────────────
        kpt_to_vert_dists = torch.cdist(trgt_kpts.float(), trgt_verts.float())  # (B, K, V)
        if trgt_verts_mask is not None:
            kpt_to_vert_dists = kpt_to_vert_dists.masked_fill(
                ~trgt_verts_mask.unsqueeze(1), float("inf")
            )
        nn_trgt_vert = kpt_to_vert_dists.argmin(dim=2)  # (B, K)

        # ── 2. gather trgt vertex features ────────────────────────────────────
        B_idx = torch.arange(B, device=dev)[:, None].expand(B, K)  # (B, K)
        trgt_kpt_feats = trgt_feats[B_idx, nn_trgt_vert]  # (B, K, F)

        # ── 3. nearest src vertex by feature distance ──────────────────────────
        feat_dists = torch.cdist(trgt_kpt_feats.float(), src_feats.float())  # (B, K, V)
        if src_verts_mask is not None:
            feat_dists = feat_dists.masked_fill(
                ~src_verts_mask.unsqueeze(1), float("inf")
            )
        nn_src_vert = feat_dists.argmin(dim=2)  # (B, K)

        # ── 4. predicted src position ──────────────────────────────────────────
        pred_src = src_verts[B_idx, nn_src_vert]  # (B, K, 3)

        # ── euclidean error vs GT src keypoint positions ───────────────────────
        kpts_euc_dist = (pred_src - src_kpts.float()).norm(dim=-1)  # (B, K)

        # ── valid-kpt mask ─────────────────────────────────────────────────────
        if src_mask is not None and trgt_mask is not None:
            valid = src_mask & trgt_mask
        elif src_mask is not None:
            valid = src_mask
        elif trgt_mask is not None:
            valid = trgt_mask
        else:
            valid = torch.ones(B, K, dtype=torch.bool, device=dev)

        n_valid = valid.float().sum(dim=1).clamp(min=1)  # (B,)

        # ── PCK01: error < 0.1 * largest bounding-box dim of target ───────────
        if trgt_verts_mask is not None:
            # exclude padding vertices from bounding box
            mask3d = trgt_verts_mask.unsqueeze(-1).expand_as(trgt_verts)
            tv = trgt_verts.float()
            v_max = tv.masked_fill(~mask3d, float("-inf")).amax(dim=1)
            v_min = tv.masked_fill(~mask3d, float("inf")).amin(dim=1)
            trgt_max_dim = (v_max - v_min).amax(dim=1)  # (B,)
        else:
            trgt_max_dim = (
                trgt_verts.float().amax(dim=1) - trgt_verts.float().amin(dim=1)
            ).amax(dim=1)  # (B,)
        threshold = 0.1 * trgt_max_dim  # (B,)

        pck01 = (
            ((kpts_euc_dist < threshold.unsqueeze(1)).float() * valid.float()).sum(dim=1)
            / n_valid
        )  # (B,)

        # ── mean euclidean error over valid kpts ───────────────────────────────
        geo_dist_mean = (kpts_euc_dist * valid.float()).sum(dim=1) / n_valid  # (B,)

        # ── qualitative: render correspondence images ──────────────────────────
        qualit_imgs = _render_corr_imgs(
            src_verts, trgt_verts,
            src_kpts, trgt_kpts, pred_src,
            valid,
            src_mask=src_verts_mask,
            trgt_mask=trgt_verts_mask,
            src_meshes=batch.src_meshes,
            trgt_meshes=batch.trgt_meshes,
        )

        quant = ObjectPairQuantBatch(
            kpts_euc_dist = kpts_euc_dist,
            kpts_mask     = valid,
            pck01         = pck01,
            geo_dist_mean = geo_dist_mean,
        )
        qualit = ObjectPairQualitBatch(
            trgt_src_vert_corr      = nn_src_vert,
            trgt_src_vert_corr_mask = valid,
            imgs                    = qualit_imgs,
        )
        return quant, qualit

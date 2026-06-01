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
    H: int = 256,
    W: int = 256,
) -> Optional[Tensor]:
    """Return (B, 3, H, 2*W) float32 [0,1] images.

    Left panel: target object point cloud with target keypoints (filled circles).
    Right panel: source object point cloud with GT src kpts (open rings) and
                 predicted src positions (filled circles, same colour as trgt kpt).
    Coloured lines cross the centre separator to connect each target keypoint to
    its predicted source position, visualising correspondence quality.
    """
    try:
        import numpy as np
        import colorsys
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    B, K, _ = src_kpts.shape
    imgs_out = []
    margin = 14
    r_dot = max(4, H // 52)

    for b in range(B):
        sv = src_verts[b].float().cpu()    # (V, 3)
        tv = trgt_verts[b].float().cpu()   # (V, 3)
        sk = src_kpts[b].float().cpu()     # (K, 3)
        tk = trgt_kpts[b].float().cpu()    # (K, 3)
        pk = pred_src[b].float().cpu()     # (K, 3)
        vm = valid[b].cpu()               # (K,) bool

        # filter out padding vertices
        if src_mask is not None:
            sv = sv[src_mask[b].cpu()]
        if trgt_mask is not None:
            tv = tv[trgt_mask[b].cpu()]

        # common scale from both objects + keypoints so they are comparable
        ref_pts = torch.cat([sv, tv, sk[vm], tk[vm], pk[vm]], dim=0) if vm.any() else torch.cat([sv, tv], dim=0)
        if ref_pts.shape[0] == 0:
            imgs_out.append(torch.ones(3, H, 2 * W))
            continue
        mn = ref_pts.min(0).values
        mx = ref_pts.max(0).values
        scale = (mx - mn).max().clamp(min=1e-6).item()
        cx = ((mn[0] + mx[0]) / 2).item()
        cy = ((mn[1] + mx[1]) / 2).item()

        def to_px(pts: Tensor, x_offset: int = 0):
            p = pts.numpy()
            u = ((p[:, 0] - cx) / scale + 0.5) * (W - 2 * margin) + margin + x_offset
            v = (-(p[:, 1] - cy) / scale + 0.5) * (H - 2 * margin) + margin
            return (
                u.clip(0, 2 * W - 1).astype(int),
                v.clip(0, H - 1).astype(int),
            )

        canvas = np.full((H, 2 * W, 3), 235, dtype=np.uint8)
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)

        # vertical separator
        draw.line([(W, 0), (W, H - 1)], fill=(150, 150, 150), width=2)

        # draw vertex point clouds
        if tv.shape[0] > 0:
            us, vs = to_px(tv, x_offset=0)
            for u, v in zip(us, vs):
                draw.ellipse([(u, v), (u + 1, v + 1)], fill=(185, 185, 185))

        if sv.shape[0] > 0:
            us, vs = to_px(sv, x_offset=W)
            for u, v in zip(us, vs):
                draw.ellipse([(u, v), (u + 1, v + 1)], fill=(185, 185, 185))

        # draw correspondences
        for k in range(K):
            if not bool(vm[k]):
                continue
            rf, gf, bf = colorsys.hsv_to_rgb(k / max(K, 1), 0.88, 0.85)
            color = (int(rf * 255), int(gf * 255), int(bf * 255))

            tk_u, tk_v = to_px(tk[k: k + 1], x_offset=0)
            pk_u, pk_v = to_px(pk[k: k + 1], x_offset=W)
            sk_u, sk_v = to_px(sk[k: k + 1], x_offset=W)

            tku, tkv = int(tk_u[0]), int(tk_v[0])
            pku, pkv = int(pk_u[0]), int(pk_v[0])
            sku, skv = int(sk_u[0]), int(sk_v[0])

            # line from trgt kpt → predicted src position (crosses separator)
            draw.line([(tku, tkv), (pku, pkv)], fill=color, width=1)

            # target keypoint: filled circle
            draw.ellipse(
                [(tku - r_dot, tkv - r_dot), (tku + r_dot, tkv + r_dot)],
                fill=color, outline=(0, 0, 0),
            )
            # predicted src position: filled circle
            draw.ellipse(
                [(pku - r_dot, pkv - r_dot), (pku + r_dot, pkv + r_dot)],
                fill=color, outline=(0, 0, 0),
            )
            # GT src keypoint: open ring (outline only)
            draw.ellipse(
                [(sku - r_dot, skv - r_dot), (sku + r_dot, skv + r_dot)],
                outline=color,
            )

        arr = np.array(img_pil).astype(np.float32) / 255.0  # (H, 2W, 3)
        imgs_out.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, 2W)

    return torch.stack(imgs_out)  # (B, 3, H, 2W)


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

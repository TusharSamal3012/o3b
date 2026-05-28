from __future__ import annotations

from typing import Tuple

import torch

from od3d_basic.task.task import OD3D_Task, register_task
from od3d_basic.data.datatypes.object import ObjectPairBatch
from od3d_basic.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from od3d_basic.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


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

        quant = ObjectPairQuantBatch(
            kpts_euc_dist = kpts_euc_dist,
            kpts_mask     = valid,
            pck01         = pck01,
            geo_dist_mean = geo_dist_mean,
        )
        qualit = ObjectPairQualitBatch(
            trgt_src_vert_corr      = nn_src_vert,
            trgt_src_vert_corr_mask = valid,
        )
        return quant, qualit

from __future__ import annotations

from typing import Tuple

import torch

from o3b.task.task import OD3D_Task, register_task
from o3b.data.datatypes.object import ObjectPairBatch
from o3b.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from o3b.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


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
      3. Map the query keypoints (in query-camera space) into the target camera
         space and compare against the GT target keypoints.
      4. Euclidean error, thresholded at ``0.1 * trgt_obj_size`` for PCK.

    When predicted poses are absent the GT poses are used (oracle upper bound).
    """

    def __init__(self, **kwargs):
        pass

    def forward(self, batch: ObjectPairBatch) -> Tuple[ObjectPairQuantBatch, ObjectPairQualitBatch]:
        from o3b.cv.geometry.transform import (
            transf3d_broadcast, tform4x4_broadcast, inv_tform4x4,
        )

        src_kpts       = batch.src_obj_kpts3d          # (B, K, 3) NCDS
        trgt_kpts      = batch.trgt_obj_kpts3d         # (B, K, 3) NCDS
        src_kpts_mask  = batch.src_obj_kpts3d_mask     # (B, K) or None
        trgt_kpts_mask = batch.trgt_obj_kpts3d_mask    # (B, K) or None
        src_ncds       = batch.src_cam_tform4x4_obj_ncds   # (B, 4, 4) GT ncds→cam
        trgt_ncds      = batch.trgt_cam_tform4x4_obj_ncds  # (B, 4, 4)
        trgt_obj_size  = batch.trgt_obj_size               # (B,)

        # Predicted poses, falling back to GT (oracle) when unavailable.
        pred_src  = batch.src_pred_cam_tform4x4_obj  if batch.src_pred_cam_tform4x4_obj  is not None else batch.src_cam_tform4x4_obj
        pred_trgt = batch.trgt_pred_cam_tform4x4_obj if batch.trgt_pred_cam_tform4x4_obj is not None else batch.trgt_cam_tform4x4_obj

        required = (src_kpts, trgt_kpts, src_ncds, trgt_ncds, pred_src, pred_trgt, trgt_obj_size)
        if any(x is None for x in required):
            return ObjectPairQuantBatch(), ObjectPairQualitBatch()

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
        n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)  # (B,)

        # ── step 1: kpts (NCDS) → camera space (metric, GT poses) ─────────────
        query_kpts_cam_q = transf3d_broadcast(src_kpts.float(),  src_ncds.float().unsqueeze(1))   # (B, K, 3)
        gt_trgt_kpts_cam_t = transf3d_broadcast(trgt_kpts.float(), trgt_ncds.float().unsqueeze(1))  # (B, K, 3)

        # ── step 2: predicted relative camera transform (target ← query) ──────
        cam_query_tform4x4_cam_target = tform4x4_broadcast(pred_src.float(), inv_tform4x4(pred_trgt.float()))
        cam_target_tform4x4_cam_query = inv_tform4x4(cam_query_tform4x4_cam_target)  # (B, 4, 4)

        # ── step 3: query kpts → predicted target camera space ────────────────
        pred_trgt_kpts_cam_t = transf3d_broadcast(
            query_kpts_cam_q, cam_target_tform4x4_cam_query.unsqueeze(1),
        )  # (B, K, 3)

        # ── step 4: euclidean error + PCK @ 0.1 * trgt_obj_size ───────────────
        cam_kpts_trgt_euc_dist = (pred_trgt_kpts_cam_t - gt_trgt_kpts_cam_t).norm(dim=-1)  # (B, K)

        threshold = 0.1 * trgt_obj_size.float().to(dev)  # (B,)
        correct = (cam_kpts_trgt_euc_dist < threshold.unsqueeze(1)).float() * kpts_valid.float()

        cam_kpts_trgt_euc_dist_mean = (cam_kpts_trgt_euc_dist * kpts_valid.float()).sum(dim=1) / n_valid  # (B,)
        cam_kpts_trgt_pck01 = correct.sum(dim=1) / n_valid  # (B,)

        quant = ObjectPairQuantBatch(
            kpts_trgt_euc_dist = cam_kpts_trgt_euc_dist,
            kpts_mask          = kpts_valid,
            extra = dict(
                cam_kpts_trgt_euc_dist_mean = cam_kpts_trgt_euc_dist_mean,
                cam_kpts_trgt_pck01         = cam_kpts_trgt_pck01,
            ),
        )
        return quant, ObjectPairQualitBatch()

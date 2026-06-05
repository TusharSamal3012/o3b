from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from od3d_basic.task.task import OD3D_Task, register_task
from od3d_basic.data.datatypes.object import ObjectPairBatch
from od3d_basic.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from od3d_basic.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


# ── geodesic helper ───────────────────────────────────────────────────────────

def _geodesic_pairwise(mesh, src_indices, dst_indices) -> "np.ndarray":
    """Return geodesic distance for each (src_indices[i], dst_indices[i]) pair.

    Uses torch_geometric.utils.geodesic_distance (heat-method).
    Returns a float32 numpy array of shape (N,).
    """
    import numpy as np
    from torch_geometric.utils import geodesic_distance

    pos  = mesh.verts.float().cpu()                      # (V, 3)
    face = mesh.faces.t().contiguous().long().cpu()      # (3, F)
    src  = torch.tensor(src_indices, dtype=torch.long)
    dest = torch.tensor(dst_indices, dtype=torch.long)
    dists = geodesic_distance(pos, face, src=src, dest=dest, norm=True)
    return dists.cpu().numpy().astype(np.float32)


# ── part correspondence rendering ─────────────────────────────────────────────

def _render_part_imgs(
    src_feats:      "Tensor",             # (B, V_src, F)
    trgt_feats:     "Tensor",             # (B, V_trgt, F)
    src_verts_mask:  "Optional[Tensor]",  # (B, V_src)  bool or None
    trgt_verts_mask: "Optional[Tensor]",  # (B, V_trgt) bool or None
    src_part_id:    "Tensor",             # (B, V_src) int64
    src_meshes:     list,
    trgt_meshes:    list,
    H: int = 256,
    W: int = 256,
) -> "Optional[Tensor]":
    """Return (B, 3, 2H, 2W) images.

    Left panel:  source mesh colored by source part label.
    Right panel: target mesh where each vertex is colored by the part label of
                 its nearest source vertex in feature space.

    Both panels use the same HSV color scheme (built from the source part IDs)
    so the same part always has the same color across source and target.
    """
    try:
        import numpy as np
        from dataclasses import replace as _dc_replace
        from od3d_basic.io import _mesh_to_trimesh
        from od3d_basic.cv.visual.show import (
            render_trimesh_to_tensor,
            get_default_camera_intrinsics_from_img_size,
        )
        from od3d_basic.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
        from od3d_basic.data.datatypes.object import _part_id_to_vert_colors
    except ImportError:
        return None

    B = src_feats.shape[0]
    imgs_out = []

    cam_ts = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=2, dist=5.0).float()
    cam_k  = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()

    def render_panel(mesh, cam_t):
        if mesh is None:
            return np.full((H, W, 3), 235, dtype=np.uint8)
        try:
            rgb, _ = render_trimesh_to_tensor(
                _mesh_to_trimesh(mesh), cam_k, cam_t, H, W,
                rgb_bg=[0.92, 0.92, 0.92],
            )
            return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            return np.full((H, W, 3), 235, dtype=np.uint8)

    for b in range(B):
        sm = src_meshes[b]
        tm = trgt_meshes[b]
        if sm is None or tm is None:
            continue

        # Real vertex counts from the mesh objects — the batch tensors may be
        # padded to V_max which is larger; trimesh requires len(vert_colors) == len(verts).
        V_src_real  = sm.verts.shape[0]
        V_trgt_real = tm.verts.shape[0]

        src_pid_b = src_part_id[b, :V_src_real].cpu()  # (V_src_real,)

        # Source: direct part-label coloring (builds the canonical HSV color map)
        src_colors = _part_id_to_vert_colors(src_pid_b)

        # Target: each vertex gets the color of its nearest source vertex's part
        sf = src_feats[b, :V_src_real].float()    # (V_src_real, F)
        tf = trgt_feats[b, :V_trgt_real].float()  # (V_trgt_real, F)
        feat_dists = torch.cdist(tf, sf)           # (V_trgt_real, V_src_real)
        if src_verts_mask is not None:
            feat_dists = feat_dists.masked_fill(~src_verts_mask[b, :V_src_real].unsqueeze(0), float("inf"))
        nn_src = feat_dists.argmin(dim=1).cpu()    # (V_trgt_real,)
        trgt_mapped_part = src_pid_b[nn_src]       # (V_trgt_real,) — mapped source part IDs
        # Use src_pid_b as reference so the hue map matches the source panel
        trgt_colors = _part_id_to_vert_colors(trgt_mapped_part, reference_part_id=src_pid_b)

        src_colored  = _dc_replace(sm, vert_colors=src_colors)
        trgt_colored = _dc_replace(tm, vert_colors=trgt_colors)

        rows = []
        for v_idx in range(cam_ts.shape[0]):
            cam_t   = cam_ts[v_idx]
            src_pnl = render_panel(src_colored,  cam_t)
            trg_pnl = render_panel(trgt_colored, cam_t)
            rows.append(np.concatenate([src_pnl, trg_pnl], axis=1))

        canvas = np.concatenate(rows, axis=0).astype(np.float32) / 255.0
        imgs_out.append(torch.from_numpy(canvas).permute(2, 0, 1))

    return torch.stack(imgs_out) if imgs_out else None


# ── qualitative rendering ─────────────────────────────────────────────────────

def _render_corr_imgs(
    src_verts:      Tensor,           # (B, V_src, 3)
    trgt_verts:     Tensor,           # (B, V_trgt, 3)
    trgt_kpts:      Tensor,           # (B, K, 3)  GT target keypoints
    pred_src_kpts:  Tensor,           # (B, K, 3)  predicted source positions
    trgt_euc_dist:  Tensor,           # (B, K)     target-space Euclidean error
    trgt_max_dim:   Tensor,           # (B,)       target bounding-box size
    kpts_valid:     Tensor,           # (B, K) bool
    src_verts_mask:  Optional[Tensor],
    trgt_verts_mask: Optional[Tensor],
    src_meshes:  Optional[list] = None,
    trgt_meshes: Optional[list] = None,
    H: int = 256,
    W: int = 256,
) -> Optional[Tensor]:
    """Return (B, 3, 2H, 2W) float32 [0,1] correspondence images.

    Left panel:  target mesh with GT target keypoints (filled circles).
    Right panel: source mesh with predicted source positions (filled circles).
    Lines connect GT target keypoint to its predicted source position.
    Color: green = correct (target-space Euclidean error < 0.1 * trgt_max_dim),
           red   = wrong.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        from od3d_basic.io import _mesh_to_trimesh
        from od3d_basic.cv.visual.show import (
            render_trimesh_to_tensor,
            get_default_camera_intrinsics_from_img_size,
        )
        from od3d_basic.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
    except ImportError:
        return None

    B, K, _ = trgt_kpts.shape
    imgs_out = []
    r_dot = max(4, H // 52)

    cam_ts = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=2, dist=5.0).float()
    cam_k  = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()
    fx  = cam_k[0, 0].item()
    fy  = cam_k[1, 1].item()
    cx_k = cam_k[0, 2].item()
    cy_k = cam_k[1, 2].item()

    def project(pts_3d: Tensor, cam_t: Tensor):
        pts_h   = torch.cat([pts_3d.float().cpu(), torch.ones(len(pts_3d), 1)], dim=1)
        pts_cam = (cam_t @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2].clamp(min=1e-6).numpy()
        u = fx * pts_cam[:, 0].numpy() / z + cx_k
        v = fy * pts_cam[:, 1].numpy() / z + cy_k
        return u, v

    def render_panel(mesh, cam_t: Tensor) -> "np.ndarray":
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
        tk  = trgt_kpts[b].float().cpu()
        psk = pred_src_kpts[b].float().cpu()
        vm  = kpts_valid[b].cpu()
        threshold = 0.1 * trgt_max_dim[b].item()
        is_correct = trgt_euc_dist[b].cpu() < threshold  # (K,)

        sm = src_meshes[b]  if src_meshes  is not None else None
        tm = trgt_meshes[b] if trgt_meshes is not None else None

        rows = []
        for v_idx in range(cam_ts.shape[0]):
            cam_t = cam_ts[v_idx]

            trgt_panel = render_panel(tm, cam_t)
            src_panel  = render_panel(sm, cam_t)

            row_arr = np.concatenate([trgt_panel, src_panel], axis=1)
            row_pil = Image.fromarray(row_arr)
            draw = ImageDraw.Draw(row_pil)
            draw.line([(W, 0), (W, H - 1)], fill=(120, 120, 120), width=2)

            tk_u,  tk_v  = project(tk,  cam_t)
            psk_u, psk_v = project(psk, cam_t)

            for k_i in range(K):
                if not bool(vm[k_i]):
                    continue
                color = (0, 180, 0) if bool(is_correct[k_i]) else (0, 0, 200)

                tku  = int(np.clip(tk_u[k_i],  0, W - 1))
                tkv  = int(np.clip(tk_v[k_i],  0, H - 1))
                psku = int(np.clip(psk_u[k_i], 0, W - 1)) + W
                pskv = int(np.clip(psk_v[k_i], 0, H - 1))

                draw.line([(tku, tkv), (psku, pskv)], fill=color, width=1)
                draw.ellipse(
                    [(tku - r_dot, tkv - r_dot), (tku + r_dot, tkv + r_dot)],
                    fill=color, outline=(0, 0, 0),
                )
                draw.ellipse(
                    [(psku - r_dot, pskv - r_dot), (psku + r_dot, pskv + r_dot)],
                    fill=color, outline=(0, 0, 0),
                )

            rows.append(np.array(row_pil))

        canvas = np.concatenate(rows, axis=0)
        arr = canvas.astype(np.float32) / 255.0
        imgs_out.append(torch.from_numpy(arr).permute(2, 0, 1))

    return torch.stack(imgs_out)  # (B, 3, 2H, 2W)


# ── task ──────────────────────────────────────────────────────────────────────

@register_task("Crsp3DNNTask")
class Crsp3DNNTask(OD3D_Task):
    """Feature-based nearest-neighbour 3D correspondence.

    Pipeline (same for keypoints and parts):
      1. target query vertex  →  nearest source vertex  (feature space)
      2. predicted source vertex  →  nearest target vertex  (3D Euclidean, round-trip)
      3. measure Euclidean and geodesic error on the TARGET mesh

    Keypoint metric:
      query = nearest target vertex to each GT target keypoint (3D)
      error  = distance between predicted round-trip target vertex and GT target keypoint

    Part metric:
      query = each target vertex that has a valid part label (part_id >= 0)
      error  = distance between predicted round-trip target vertex and the query vertex
      PCK    = fraction of queries where the predicted target vertex shares the same part label
    """

    def __init__(self, **kwargs):
        pass

    def forward(
        self,
        batch: ObjectPairBatch,
    ) -> Tuple[ObjectPairQuantBatch, ObjectPairQualitBatch]:

        src_kpts      = batch.src_obj_kpts3d       # (B, K, 3)
        trgt_kpts     = batch.trgt_obj_kpts3d      # (B, K, 3)
        src_kpts_mask  = batch.src_obj_kpts3d_mask  # (B, K) bool or None
        trgt_kpts_mask = batch.trgt_obj_kpts3d_mask # (B, K) bool or None
        src_verts     = batch.src_verts3d           # (B, V_src, 3)
        src_feats     = batch.src_verts3d_feats     # (B, V_src, F)
        trgt_verts    = batch.trgt_verts3d          # (B, V_trgt, 3)
        trgt_feats    = batch.trgt_verts3d_feats    # (B, V_trgt, F)
        trgt_verts_mask = batch.trgt_verts3d_feats_mask  # (B, V_trgt) bool or None
        src_verts_mask  = batch.src_verts3d_feats_mask   # (B, V_src)  bool or None
        trgt_part_id    = batch.trgt_verts3d_part_id     # (B, V_trgt) int64 or None
        src_part_id     = batch.src_verts3d_part_id      # (B, V_src)  int64 or None

        has_kpts  = all(x is not None for x in (src_kpts, trgt_kpts, src_verts, src_feats, trgt_verts, trgt_feats))
        has_parts = trgt_part_id is not None and all(x is not None for x in (src_verts, src_feats, trgt_verts, trgt_feats))
        has_part_viz = (src_part_id is not None and src_feats is not None and trgt_feats is not None
                        and batch.src_meshes is not None and batch.trgt_meshes is not None)

        if not has_kpts and not has_parts:
            return ObjectPairQuantBatch(), ObjectPairQualitBatch()

        B = (src_verts if src_verts is not None else trgt_verts).shape[0]
        dev = (src_verts if src_verts is not None else trgt_verts).device

        # ── target bounding-box size (used for PCK threshold) ─────────────────
        if trgt_verts is not None:
            if trgt_verts_mask is not None:
                mask3d = trgt_verts_mask.unsqueeze(-1).expand_as(trgt_verts)
                tv = trgt_verts.float()
                v_max = tv.masked_fill(~mask3d, float("-inf")).amax(dim=1)
                v_min = tv.masked_fill(~mask3d, float("inf")).amin(dim=1)
                trgt_max_dim = (v_max - v_min).amax(dim=1)  # (B,)
            else:
                trgt_max_dim = (trgt_verts.float().amax(dim=1) - trgt_verts.float().amin(dim=1)).amax(dim=1)
        else:
            trgt_max_dim = torch.ones(B, device=dev)

        # ── helper: apply vertex mask to feature distance matrix ──────────────
        def _mask_feat_dists(dists, src_mask):
            if src_mask is not None:
                dists = dists.masked_fill(~src_mask.unsqueeze(1), float("inf"))
            return dists

        def _mask_vert_dists(dists, trgt_mask):
            if trgt_mask is not None:
                dists = dists.masked_fill(~trgt_mask.unsqueeze(1), float("inf"))
            return dists

        # ═══════════════════════════════════════════════════════════════════════
        # KEYPOINT CORRESPONDENCES
        # ═══════════════════════════════════════════════════════════════════════
        quant_kpts = {}
        pred_src_kpt_pos = None  # kept for qualitative rendering

        if has_kpts:
            B_k, K, _ = src_kpts.shape
            B_idx_k   = torch.arange(B_k, device=dev)[:, None].expand(B_k, K)

            # valid keypoint mask
            if src_kpts_mask is not None and trgt_kpts_mask is not None:
                kpts_valid = src_kpts_mask & trgt_kpts_mask
            elif src_kpts_mask is not None:
                kpts_valid = src_kpts_mask
            elif trgt_kpts_mask is not None:
                kpts_valid = trgt_kpts_mask
            else:
                kpts_valid = torch.ones(B_k, K, dtype=torch.bool, device=dev)

            n_kpts_valid = kpts_valid.float().sum(dim=1).clamp(min=1)  # (B,)

            # Step 1: trgt_kpt → nearest trgt_vert (3D Euclidean)
            kpt_to_trgt_vert_dists = torch.cdist(trgt_kpts.float(), trgt_verts.float())  # (B, K, V_trgt)
            kpt_to_trgt_vert_dists = _mask_vert_dists(kpt_to_trgt_vert_dists, trgt_verts_mask)
            nn_trgt_kpt_vert = kpt_to_trgt_vert_dists.argmin(dim=2)  # (B, K)

            # Step 2: trgt_vert feature → nearest src_vert (feature space)
            trgt_kpt_feats = trgt_feats[B_idx_k, nn_trgt_kpt_vert]  # (B, K, F)
            kpt_feat_dists = torch.cdist(trgt_kpt_feats.float(), src_feats.float())  # (B, K, V_src)
            kpt_feat_dists = _mask_feat_dists(kpt_feat_dists, src_verts_mask)
            nn_src_kpt_vert = kpt_feat_dists.argmin(dim=2)  # (B, K)

            # Step 3: src_vert position → nearest trgt_vert (3D, round-trip)
            pred_src_kpt_pos    = src_verts[B_idx_k, nn_src_kpt_vert]    # (B, K, 3)  for qualitative viz
            pred_trgt_kpt_dists = torch.cdist(pred_src_kpt_pos.float(), trgt_verts.float())  # (B, K, V_trgt)
            pred_trgt_kpt_dists = _mask_vert_dists(pred_trgt_kpt_dists, trgt_verts_mask)
            pred_trgt_kpt_vert  = pred_trgt_kpt_dists.argmin(dim=2)      # (B, K)
            pred_trgt_kpt_pos   = trgt_verts[B_idx_k, pred_trgt_kpt_vert]  # (B, K, 3)

            # Step 4: Euclidean error on TARGET mesh
            kpts_trgt_euc_dist = (pred_trgt_kpt_pos - trgt_kpts.float()).norm(dim=-1)  # (B, K)

            # Step 5: PCK @ 0.1 * trgt_max_dim (target-space)
            kpts_trgt_pck01 = (
                ((kpts_trgt_euc_dist < trgt_max_dim.unsqueeze(1) * 0.1).float() * kpts_valid.float()).sum(dim=1)
                / n_kpts_valid
            )  # (B,)

            kpts_trgt_euc_dist_mean = (kpts_trgt_euc_dist * kpts_valid.float()).sum(dim=1) / n_kpts_valid  # (B,)

            # Step 6: geodesic error on TARGET mesh (torch_geometric heat method)
            kpts_trgt_geo_dist = torch.zeros(B_k, K, dtype=torch.float32)
            if batch.trgt_meshes is not None:
                import numpy as np
                for b in range(B_k):
                    mesh = batch.trgt_meshes[b]
                    if mesh is None:
                        continue
                    valid_k      = kpts_valid[b].cpu().numpy()
                    pred_verts_b = pred_trgt_kpt_vert[b].cpu().numpy()
                    gt_verts_b   = nn_trgt_kpt_vert[b].cpu().numpy()
                    valid_idx    = np.where(valid_k)[0]
                    if len(valid_idx) == 0:
                        continue
                    geo = _geodesic_pairwise(
                        mesh,
                        pred_verts_b[valid_idx],
                        gt_verts_b[valid_idx],
                    )  # (n_valid,)
                    for j, k in enumerate(valid_idx):
                        d = float(geo[j])
                        kpts_trgt_geo_dist[b, k] = d if np.isfinite(d) else 0.0

            geo_t = kpts_trgt_geo_dist.to(dev)  # (B, K)
            kpts_trgt_geo_dist_mean = (
                (geo_t * kpts_valid.float()).sum(dim=1) / n_kpts_valid
            )  # (B,)

            # AUC: integrate PCK curve over linspace thresholds in [0, 1]
            thresholds = torch.linspace(0, 1, 100, device=dev)  # (T,)
            pck_per_thresh = (
                (geo_t.unsqueeze(-1) < thresholds.unsqueeze(0).unsqueeze(0)).float()  # (B, K, T)
                * kpts_valid.float().unsqueeze(-1)                                     # (B, K, 1)
            ).sum(dim=1) / n_kpts_valid.unsqueeze(-1)                                  # (B, T)
            kpts_trgt_geo_auc = pck_per_thresh.mean(dim=1)  # (B,)

            quant_kpts = dict(
                kpts_trgt_euc_dist      = kpts_trgt_euc_dist,
                kpts_trgt_geo_dist      = geo_t,
                kpts_mask               = kpts_valid,
                kpts_trgt_pck01         = kpts_trgt_pck01,
                kpts_trgt_euc_dist_mean = kpts_trgt_euc_dist_mean,
                kpts_trgt_geo_dist_mean = kpts_trgt_geo_dist_mean,
                kpts_trgt_geo_auc       = kpts_trgt_geo_auc,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # PART CORRESPONDENCES
        # ═══════════════════════════════════════════════════════════════════════
        quant_parts = {}

        if has_parts:
            import numpy as np

            parts_trgt_euc_dist_mean_list = []
            parts_trgt_geo_dist_mean_list = []
            parts_trgt_geo_dist_raw_list  = []  # raw (V_part,) tensors for AUC
            parts_trgt_pck_list           = []

            for b in range(B):
                part_id_b = trgt_part_id[b]  # (V_trgt,) int64

                # mask: real vertex with valid part label
                valid_part_mask = part_id_b >= 0
                if trgt_verts_mask is not None:
                    valid_part_mask = valid_part_mask & trgt_verts_mask[b]
                valid_part_idx = valid_part_mask.nonzero(as_tuple=True)[0]  # (V_part,)

                if valid_part_idx.numel() == 0:
                    parts_trgt_euc_dist_mean_list.append(torch.tensor(0.0))
                    parts_trgt_geo_dist_mean_list.append(torch.tensor(float("nan")))
                    parts_trgt_geo_dist_raw_list.append(None)
                    parts_trgt_pck_list.append(torch.tensor(0.0))
                    continue

                trgt_verts_b = trgt_verts[b].float()   # (V_trgt, 3)
                trgt_feats_b = trgt_feats[b].float()   # (V_trgt, F)
                src_verts_b  = src_verts[b].float()    # (V_src, 3)
                src_feats_b  = src_feats[b].float()    # (V_src, F)

                query_verts = trgt_verts_b[valid_part_idx]  # (V_part, 3)
                query_feats = trgt_feats_b[valid_part_idx]  # (V_part, F)
                query_parts = part_id_b[valid_part_idx]     # (V_part,) int64

                # Step 1: query_vert feature → nearest src_vert (feature space)
                part_feat_dists = torch.cdist(query_feats, src_feats_b)  # (V_part, V_src)
                if src_verts_mask is not None:
                    part_feat_dists = part_feat_dists.masked_fill(~src_verts_mask[b].unsqueeze(0), float("inf"))
                nn_src_part_vert = part_feat_dists.argmin(dim=1)  # (V_part,)

                # Step 2: src_vert position → nearest trgt_vert (3D, round-trip)
                pred_src_part_pos   = src_verts_b[nn_src_part_vert]   # (V_part, 3)
                pred_trgt_part_dists = torch.cdist(pred_src_part_pos, trgt_verts_b)  # (V_part, V_trgt)
                if trgt_verts_mask is not None:
                    pred_trgt_part_dists = pred_trgt_part_dists.masked_fill(~trgt_verts_mask[b].unsqueeze(0), float("inf"))
                pred_trgt_part_vert = pred_trgt_part_dists.argmin(dim=1)  # (V_part,)
                pred_trgt_part_pos  = trgt_verts_b[pred_trgt_part_vert]   # (V_part, 3)

                # Euclidean error on TARGET mesh
                part_euc_dist = (pred_trgt_part_pos - query_verts).norm(dim=-1)  # (V_part,)
                parts_trgt_euc_dist_mean_list.append(part_euc_dist.mean())

                # Part PCK: predicted target vertex has the same part label
                pred_part_id  = part_id_b[pred_trgt_part_vert]  # (V_part,)
                part_correct  = (pred_part_id == query_parts).float()
                parts_trgt_pck_list.append(part_correct.mean())

                # Geodesic error on TARGET mesh (torch_geometric heat method)
                if batch.trgt_meshes is not None and batch.trgt_meshes[b] is not None:
                    geo_dists_np = _geodesic_pairwise(
                        batch.trgt_meshes[b],
                        pred_trgt_part_vert.cpu().numpy(),
                        valid_part_idx.cpu().numpy(),
                    )  # (V_part,) float32
                    geo_dists_t  = torch.from_numpy(geo_dists_np)
                    finite       = geo_dists_t[torch.isfinite(geo_dists_t)]
                    geo_mean     = finite.mean() if len(finite) > 0 else torch.tensor(float("nan"))
                    parts_trgt_geo_dist_mean_list.append(geo_mean)
                    parts_trgt_geo_dist_raw_list.append(finite if len(finite) > 0 else None)
                else:
                    parts_trgt_geo_dist_mean_list.append(torch.tensor(float("nan")))
                    parts_trgt_geo_dist_raw_list.append(None)

            # AUC per sample from the per-sample geodesic lists
            thresholds_parts = torch.linspace(0, 1, 100)  # (T,) on CPU
            parts_trgt_geo_auc_list = []
            for geo_mean_t, geo_dists_t in zip(
                parts_trgt_geo_dist_mean_list,
                parts_trgt_geo_dist_raw_list,
            ):
                if geo_dists_t is None:
                    parts_trgt_geo_auc_list.append(torch.tensor(float("nan")))
                else:
                    pck = (geo_dists_t.unsqueeze(0) < thresholds_parts.unsqueeze(1)).float().mean(dim=1)
                    parts_trgt_geo_auc_list.append(pck.mean())

            quant_parts = dict(
                parts_trgt_euc_dist_mean = torch.stack(parts_trgt_euc_dist_mean_list).to(dev),
                parts_trgt_geo_dist_mean = torch.stack(parts_trgt_geo_dist_mean_list).to(dev),
                parts_trgt_geo_auc       = torch.stack(parts_trgt_geo_auc_list).to(dev),
                parts_trgt_pck           = torch.stack(parts_trgt_pck_list).to(dev),
            )

        # ── qualitative: keypoint correspondence images ───────────────────────
        qualit_imgs = None
        if has_kpts and pred_src_kpt_pos is not None:
            qualit_imgs = _render_corr_imgs(
                src_verts, trgt_verts,
                trgt_kpts, pred_src_kpt_pos,
                kpts_trgt_euc_dist, trgt_max_dim,
                kpts_valid,
                src_verts_mask  = src_verts_mask,
                trgt_verts_mask = trgt_verts_mask,
                src_meshes  = batch.src_meshes,
                trgt_meshes = batch.trgt_meshes,
            )

        # ── qualitative: part correspondence images ───────────────────────────
        part_imgs = None
        if has_part_viz:
            part_imgs = _render_part_imgs(
                src_feats, trgt_feats,
                src_verts_mask, trgt_verts_mask,
                src_part_id,
                batch.src_meshes, batch.trgt_meshes,
            )

        quant = ObjectPairQuantBatch(**quant_kpts, **quant_parts)
        qualit = ObjectPairQualitBatch(imgs=qualit_imgs, part_imgs=part_imgs)
        if has_kpts:
            qualit.trgt_src_vert_corr      = nn_src_kpt_vert
            qualit.trgt_src_vert_corr_mask = kpts_valid
        return quant, qualit

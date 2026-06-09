from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from o3b.data.datatypes.mesh import Mesh
from o3b.data.datatypes.frame import _stack_field, _pad_stack_field


def _draw_kpts2d_on_imgs(
    imgs: "Tensor",           # (B, 3, H, W) float32 [0,1]
    kpts2d: "Tensor",         # (B, K, 2) float32  [u, v] pixel coords
    mask: "Optional[Tensor]", # (K,) bool or None
    radius: int = 5,
) -> "Tensor":
    """Draw HSV-coloured filled circles at keypoint locations in-place (cloned copy)."""
    import colorsys
    B, _, H, W = imgs.shape
    K = kpts2d.shape[1]
    result = imgs.clone()
    ys = torch.arange(H, dtype=torch.float32, device=imgs.device)
    xs = torch.arange(W, dtype=torch.float32, device=imgs.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    for k in range(K):
        if mask is not None and not mask[k]:
            continue
        r, g, b = colorsys.hsv_to_rgb(k / max(K, 1), 0.9, 0.88)
        color = torch.tensor([r, g, b], dtype=torch.float32, device=imgs.device)
        for b_i in range(B):
            u, v = kpts2d[b_i, k, 0], kpts2d[b_i, k, 1]
            circle = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2  # (H, W)
            result[b_i, :, circle] = color[:, None]
    return result


def _render_mesh_trimesh(mesh: "Mesh", n_views: int = 6, H: int = 256, W: int = 256) -> "Tensor":
    """Render mesh from n_views azimuth angles using trimesh's offscreen backend."""
    import math
    import numpy as np
    from o3b.io import _mesh_to_trimesh
    import trimesh as tm

    mesh_tm = _mesh_to_trimesh(mesh)
    bounds  = mesh_tm.bounding_box.bounds
    center  = (bounds[0] + bounds[1]) / 2.0
    radius  = float(np.linalg.norm(bounds[1] - bounds[0])) * 0.75

    imgs = []
    for i in range(n_views):
        angle = i * 2.0 * math.pi / n_views
        eye   = center + radius * np.array([math.sin(angle), 0.3, math.cos(angle)])
        # look-at in OpenGL convention (camera looks along -Z)
        fwd   = center - eye;        fwd   /= np.linalg.norm(fwd)   + 1e-8
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right) + 1e-8
        up    = np.cross(right, fwd)
        T = np.eye(4)
        T[:3, 0] = right;  T[:3, 1] = up;  T[:3, 2] = -fwd;  T[:3, 3] = eye

        scene = tm.Scene(mesh_tm)
        scene.camera_transform = T
        try:
            data = scene.save_image(resolution=[W, H])
            img  = torch.from_numpy(
                np.frombuffer(data, dtype=np.uint8).copy().reshape(H, W, 4)[..., :3]
            ).float().permute(2, 0, 1) / 255.0
        except Exception:
            img = torch.zeros(3, H, W)
        imgs.append(img)
    return torch.stack(imgs)  # (N, 3, H, W)


def _ncds0c_vert_colors(verts: "Tensor") -> "Tensor":
    """Map (V, 3) NOCS-0c verts directly to RGB (V, 3) in [0, 1]."""
    from o3b.cv.visual.point3d_to_color3d import nocs_0c_to_rgb
    return nocs_0c_to_rgb(verts.float().cpu())


def _add_visibility_gui(server, label: str, handle_dicts: "list[dict]") -> object:
    """Add a GUI folder with one checkbox per modality key controlling all objects.

    handle_dicts: list of per-object dicts, each mapping modality key → scene handle.
    """
    KEYS = [
        ("mesh",               "Mesh"),
        ("mesh_feats",         "Mesh Feats"),
        ("mesh_parts",         "Mesh Parts"),
        ("mesh_ncds0c_3dnn",   "Mesh NCDS0C 3DNN"),
        ("mesh_ncds0c_featnn", "Mesh NCDS0C FeatNN"),
        ("pts3d",              "Points"),
        ("kpts3d",             "Keypoints"),
    ]
    gui_folder = server.gui.add_folder(label)
    with gui_folder:
        for key, label_text in KEYS:
            all_h = [d[key] for d in handle_dicts if d.get(key) is not None]
            cb = server.gui.add_checkbox(label_text, initial_value=(key == "mesh" and bool(all_h)))

            def _make_cb(cb_ref, h_refs):
                @cb_ref.on_update
                def _(_):
                    for h in h_refs:
                        h.visible = cb_ref.value
            _make_cb(cb, all_h)
    return gui_folder


def _part_id_to_vert_colors(
    part_id: "Tensor",
    reference_part_id: "Optional[Tensor]" = None,
) -> "Tensor":
    """Map (V,) int64 part IDs → (V, 3) float32 RGB in [0, 1]. ID -1 → gray.

    reference_part_id: if given, the hue mapping is built from its unique parts
    instead of from part_id.  Pass the source part_id when coloring both source
    and target so that the same part always gets the same color.
    """
    import colorsys
    ref = reference_part_id if reference_part_id is not None else part_id
    unique_parts = sorted(int(p) for p in ref.unique().tolist() if p >= 0)
    n_parts = len(unique_parts)
    part_to_color: dict = {}
    for i, pid in enumerate(unique_parts):
        r, g, b = colorsys.hsv_to_rgb(i / max(n_parts, 1), 0.85, 0.92)
        part_to_color[pid] = (r, g, b)
    colors = torch.full((part_id.shape[0], 3), 0.55, dtype=torch.float32)
    for pid, (r, g, b) in part_to_color.items():
        mask = part_id == pid
        if mask.any():
            colors[mask] = torch.tensor([r, g, b], dtype=torch.float32)
    return colors


def _pca_vert_colors(vert_feats: "Tensor") -> "Tensor":
    """Project per-vertex features (V, F) → RGB via PCA, float32 (V, 3) in [0, 1]."""
    V, F = vert_feats.shape
    flat = vert_feats.float().cpu()
    flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat = flat - flat.mean(0)
    q = min(3, F)
    _, _, Vmat = torch.pca_lowrank(flat, q=q, niter=4)
    pca = flat @ Vmat[:, :q]  # (V, q)
    if q < 3:
        pca = torch.cat([pca, torch.zeros(V, 3 - q)], dim=1)
    for c in range(3):
        ch = pca[:, c]
        pca[:, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return pca.clamp(0, 1)


@dataclass(kw_only=True)
class Object:
    object_id:               str
    pts3d:                   Optional[Tensor] = None  # (N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (N, F) or (N, V, F) multi-view
    pts3d_feats_mask:        Optional[Tensor] = None  # (N,) or (N, V) bool
    verts3d:                 Optional[Tensor] = None  # (V, 3)
    verts3d_feats:           Optional[Tensor] = None  # (V, F) or (V, V, F) multi-view
    verts3d_feats_mask:      Optional[Tensor] = None  # (V,) or (V, V) bool
    mesh:                    Optional[Mesh]   = None
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (K,)  bool
    obj_verts_part_id:       Optional[Tensor] = None  # (V,)  int64, -1 = unlabeled
    category:                Optional[int]    = None
    category_id:             Optional[int]    = None
    attributes:              Optional[dict]   = None

    def transform(self, tform4x4: "Tensor") -> "Object":
        """Return a new Object with all geometric attributes transformed by tform4x4.

        tform4x4 is a (4, 4) float tensor where:
          tform4x4[:3, :3]  is the rotation matrix R
          tform4x4[:3,  3]  is the translation vector t

        Points are mapped as  p_new = R @ p + t.
        obj_ncds0c_tform4x4_obj is updated to preserve NOCS coordinates:
          tform_new = tform_old @ inv(tform4x4)
        """
        from dataclasses import replace as _dc_replace

        T = tform4x4.float().cpu()
        R = T[:3, :3]   # (3, 3)
        t = T[:3,  3]   # (3,)

        def _xfm(pts: "Optional[Tensor]") -> "Optional[Tensor]":
            if pts is None:
                return None
            return (R @ pts.float().cpu().T).T + t

        new_mesh = None
        if self.mesh is not None:
            from dataclasses import replace as _r
            new_mesh = _r(self.mesh, verts=_xfm(self.mesh.verts))

        new_tform = None
        if self.obj_ncds0c_tform4x4_obj is not None:
            new_tform = self.obj_ncds0c_tform4x4_obj.float().cpu() @ torch.linalg.inv(T)

        return _dc_replace(
            self,
            pts3d                   = _xfm(self.pts3d),
            verts3d                 = _xfm(self.verts3d),
            obj_kpts3d              = _xfm(self.obj_kpts3d),
            mesh                    = new_mesh,
            obj_ncds0c_tform4x4_obj = new_tform,
        )

    def render_modalities(
        self,
        renderer: str = "pyrender",
        n_views: int = 4,
        H: int = 256,
        W: int = 256,
    ) -> "Optional[dict]":
        """Return dict of rendered modalities {'rgb','depth','normals'}, each (N,3,H,W)."""
        if self.mesh is None:
            return None
        from o3b.data.viz import sample_uniform_viewpoints, render_mesh_from_viewpoints
        from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size

        batch = sample_uniform_viewpoints(n_views, mesh=self.mesh)
        modalities = render_mesh_from_viewpoints(batch, H=H, W=W, renderer=renderer)

        if self.obj_kpts3d is not None:
            cam_tform4x4_obj = batch.cam_tform4x4_obj  # (B, 4, 4)
            B = cam_tform4x4_obj.shape[0]
            cam_intr4x4 = batch.cam_intr4x4
            if cam_intr4x4 is None:
                cam_intr4x4 = get_default_camera_intrinsics_from_img_size(W, H).unsqueeze(0).expand(B, -1, -1)
            elif cam_intr4x4.dim() == 2:
                cam_intr4x4 = cam_intr4x4.unsqueeze(0).expand(B, -1, -1)

            kpts3d = self.obj_kpts3d.float()       # (K, 3)
            K = kpts3d.shape[0]
            kpts3d_h = torch.cat([kpts3d, torch.ones(K, 1)], dim=1)  # (K, 4)
            kpts_cam = (cam_tform4x4_obj @ kpts3d_h.T).permute(0, 2, 1)  # (B, K, 4)
            kpts_xyz = kpts_cam[..., :3]                                   # (B, K, 3)
            kpts_proj = torch.bmm(cam_intr4x4[:, :3, :3],
                                  kpts_xyz.permute(0, 2, 1)).permute(0, 2, 1)  # (B, K, 3)
            z = kpts_proj[..., 2:3].clamp(min=1e-6)
            kpts2d = kpts_proj[..., :2] / z  # (B, K, 2)  [u, v]

            modalities["rgb"] = _draw_kpts2d_on_imgs(
                modalities["rgb"], kpts2d,
                mask=self.obj_kpts3d_mask,
                radius=max(H, W) // 50,
            )

        return modalities

    def viz(
        self,
        renderer: str = "pyrender",
        n_views: int = 6,
        H: int = 256,
        W: int = 256,
        server=None,
        node_prefix: str = "/object",
        gui_label: str = "Modalities",
        position_offset: tuple = (0.0, 0.0, 0.0),
        mesh_feats_colors:       "Optional[Tensor]" = None,
        mesh_nocs_colors:        "Optional[Tensor]" = None,
        mesh_nocs_featnn_colors: "Optional[Tensor]" = None,
    ) -> "Optional[Tensor | list]":
        """Render or display the object.

        server=None — renders n_views with the chosen renderer and returns a
                      (3, H, W_total) strip tensor in [0, 1].
        server      — adds mesh, point cloud, and keypoints to the given viser
                      server; returns the list of handles for later removal.
                      node_prefix / gui_label / position_offset allow ObjectPair
                      to place src and trgt side-by-side in the same scene.
        """
        if self.mesh is None and self.pts3d is None and self.obj_kpts3d is None:
            return None

        # ── static render (server=None) ───────────────────────────────────────
        if server is None:
            if self.mesh is None:
                return None
            if renderer == "trimesh":
                imgs = _render_mesh_trimesh(self.mesh, n_views=n_views, H=H, W=W)
            else:
                imgs = self.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)["rgb"]
            return torch.cat(list(imgs.clamp(0, 1)), dim=2)  # (3, H, W_total)

        # ── populate a viser server ───────────────────────────────────────────
        handles = self._build_scene_handles(
            server, node_prefix, position_offset,
            mesh_feats_colors, mesh_nocs_colors, mesh_nocs_featnn_colors,
        )
        gui_folder = _add_visibility_gui(server, gui_label, [handles])
        return [h for h in (*handles.values(), gui_folder) if h is not None]

    def _build_scene_handles(
        self,
        server,
        node_prefix: str,
        position_offset: tuple,
        mesh_feats_colors:       "Optional[Tensor]" = None,
        mesh_nocs_colors:        "Optional[Tensor]" = None,
        mesh_nocs_featnn_colors: "Optional[Tensor]" = None,
    ) -> "dict":
        """Add scene nodes to *server* and return a handle-dict (no GUI)."""
        import numpy as np
        from dataclasses import replace as _dc_replace
        from o3b.io import _mesh_to_trimesh

        mesh_handle = None
        if self.mesh is not None:
            mesh_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh", _mesh_to_trimesh(self.mesh),
                position=position_offset,
            )

        mesh_parts_handle = None
        if self.mesh is not None and self.obj_verts_part_id is not None:
            part_colors = _part_id_to_vert_colors(self.obj_verts_part_id)
            mesh_with_parts = _dc_replace(self.mesh, vert_colors=part_colors)
            mesh_parts_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh_parts", _mesh_to_trimesh(mesh_with_parts),
                position=position_offset,
            )
            mesh_parts_handle.visible = False

        mesh_feats_handle = None
        if self.mesh is not None and (self.mesh.vert_feats is not None or mesh_feats_colors is not None):
            feat_colors = mesh_feats_colors if mesh_feats_colors is not None else _pca_vert_colors(self.mesh.vert_feats)
            mesh_with_feats = _dc_replace(self.mesh, vert_colors=feat_colors)
            mesh_feats_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh_feats", _mesh_to_trimesh(mesh_with_feats),
                position=position_offset,
            )
            mesh_feats_handle.visible = False

        mesh_nocs_handle = None
        if self.mesh is not None and mesh_nocs_colors is not None:
            mesh_with_nocs = _dc_replace(self.mesh, vert_colors=mesh_nocs_colors)
            mesh_nocs_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh_ncds0c_3dnn", _mesh_to_trimesh(mesh_with_nocs),
                position=position_offset,
            )
            mesh_nocs_handle.visible = False

        mesh_nocs_featnn_handle = None
        if self.mesh is not None and mesh_nocs_featnn_colors is not None:
            mesh_with_nocs_featnn = _dc_replace(self.mesh, vert_colors=mesh_nocs_featnn_colors)
            mesh_nocs_featnn_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh_ncds0c_featnn", _mesh_to_trimesh(mesh_with_nocs_featnn),
                position=position_offset,
            )
            mesh_nocs_featnn_handle.visible = False

        pts_handle = None
        if self.pts3d is not None:
            pts    = self.pts3d.cpu().numpy()
            colors = np.full_like(pts, 0.7)
            pts_handle = server.scene.add_point_cloud(
                f"{node_prefix}/pts3d", points=pts, colors=colors, point_size=0.005,
                position=position_offset,
            )

        kpts_handle = None
        if self.obj_kpts3d is not None:
            from o3b.data.viz import _make_kpts_spheres
            mask_np = (self.obj_kpts3d_mask.bool()
                       if self.obj_kpts3d_mask is not None
                       else torch.ones(len(self.obj_kpts3d), dtype=torch.bool)).cpu().numpy()
            kpts_mesh = _make_kpts_spheres(self.obj_kpts3d.cpu().numpy(), mask_np)
            if kpts_mesh is not None:
                kpts_handle = server.scene.add_mesh_trimesh(
                    f"{node_prefix}/kpts3d", kpts_mesh,
                    position=position_offset,
                )

        return {
            "mesh":               mesh_handle,
            "mesh_feats":         mesh_feats_handle,
            "mesh_parts":         mesh_parts_handle,
            "mesh_ncds0c_3dnn":   mesh_nocs_handle,
            "mesh_ncds0c_featnn": mesh_nocs_featnn_handle,
            "pts3d":              pts_handle,
            "kpts3d":             kpts_handle,
        }


@dataclass(kw_only=True)
class ObjectPair:
    src_object_id:  str
    trgt_object_id: str
    src_object:     Object
    trgt_object:    Object

    def render_modalities(
        self,
        renderer: str = "pyrender",
        n_views: int = 4,
        H: int = 256,
        W: int = 256,
    ) -> "Optional[dict]":
        """Return dict of modalities with src and trgt concatenated side-by-side (dim=3)."""
        import torch
        src_mods  = self.src_object.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)
        trgt_mods = self.trgt_object.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)
        if src_mods is None and trgt_mods is None:
            return None
        base = src_mods or trgt_mods
        return {
            key: torch.cat(
                [m[key] for m in (src_mods, trgt_mods) if m is not None and key in m],
                dim=2,  # concatenate along height → (N, 3, H_src + H_trgt, W)
            )
            for key in base
        }

    def viz(
        self,
        renderer: str = "pyrender",
        n_views: int = 6,
        H: int = 256,
        W: int = 256,
        server=None,
        gap: float = 0.5,
    ) -> "Optional[Tensor | list]":
        """Render or display src and trgt objects side-by-side.

        server=None — renders both objects and concatenates their strips
                      width-wise, returning (3, H, W_src + W_trgt).
        server      — adds src at origin and trgt offset along +x by the
                      src mesh x-extent plus gap; returns all handles.
        """
        if server is None:
            src_img  = self.src_object.viz(renderer=renderer, n_views=n_views, H=H, W=W)
            trgt_img = self.trgt_object.viz(renderer=renderer, n_views=n_views, H=H, W=W)
            imgs = [i for i in (src_img, trgt_img) if i is not None]
            return torch.cat(imgs, dim=2) if imgs else None  # (3, H, W_total)

        # compute x offset so trgt sits next to src
        src_mesh = self.src_object.mesh
        if src_mesh is not None:
            x = src_mesh.verts[:, 0]
            x_offset = (x.max() - x.min()).item() + gap
        else:
            x_offset = 2.0 + gap

        # joint PCA so feature colors are comparable across both objects
        src_feats  = self.src_object.mesh.vert_feats  if (self.src_object.mesh  is not None and self.src_object.mesh.vert_feats  is not None) else None
        trgt_feats = self.trgt_object.mesh.vert_feats if (self.trgt_object.mesh is not None and self.trgt_object.mesh.vert_feats is not None) else None
        src_feat_colors = trgt_feat_colors = None
        if src_feats is not None and trgt_feats is not None:
            n_src = src_feats.shape[0]
            combined_colors = _pca_vert_colors(torch.cat([src_feats.float().cpu(), trgt_feats.float().cpu()], dim=0))
            src_feat_colors  = combined_colors[:n_src]
            trgt_feat_colors = combined_colors[n_src:]
        elif src_feats is not None:
            src_feat_colors  = _pca_vert_colors(src_feats)
        elif trgt_feats is not None:
            trgt_feat_colors = _pca_vert_colors(trgt_feats)

        # NOCS coloring: src colored by position, trgt by NN from src (verts already in NOCS-0c)
        src_nocs_colors = trgt_nocs_colors = None
        src_mesh_verts  = self.src_object.mesh.verts  if self.src_object.mesh  is not None else None
        trgt_mesh_verts = self.trgt_object.mesh.verts if self.trgt_object.mesh is not None else None
        if src_mesh_verts is not None:
            src_nocs_colors = _ncds0c_vert_colors(src_mesh_verts)  # (V_src, 3)
            if trgt_mesh_verts is not None:
                nn_idx = torch.cdist(trgt_mesh_verts.float().cpu(), src_mesh_verts.float().cpu()).argmin(dim=1)
                trgt_nocs_colors = src_nocs_colors[nn_idx]         # (V_trgt, 3)

        # FeatNN: same src NOCS colors but NN determined in feature space
        src_nocs_featnn_colors = trgt_nocs_featnn_colors = None
        src_vert_feats  = self.src_object.mesh.vert_feats  if (self.src_object.mesh  is not None and self.src_object.mesh.vert_feats  is not None) else None
        trgt_vert_feats = self.trgt_object.mesh.vert_feats if (self.trgt_object.mesh is not None and self.trgt_object.mesh.vert_feats is not None) else None
        if src_nocs_colors is not None and src_vert_feats is not None:
            src_nocs_featnn_colors = src_nocs_colors  # src keeps its own NOCS colors
            if trgt_vert_feats is not None:
                nn_idx_feat = torch.cdist(trgt_vert_feats.float().cpu(), src_vert_feats.float().cpu()).argmin(dim=1)
                trgt_nocs_featnn_colors = src_nocs_colors[nn_idx_feat]  # (V_trgt, 3)

        src_handles  = self.src_object._build_scene_handles(
            server, "/src",  (0.0, 0.0, 0.0),
            src_feat_colors, src_nocs_colors, src_nocs_featnn_colors,
        )
        trgt_handles = self.trgt_object._build_scene_handles(
            server, "/trgt", (x_offset, 0.0, 0.0),
            trgt_feat_colors, trgt_nocs_colors, trgt_nocs_featnn_colors,
        )
        gui_folder = _add_visibility_gui(server, "Objects", [src_handles, trgt_handles])
        all_handles = [*src_handles.values(), *trgt_handles.values(), gui_folder]
        return [h for h in all_handles if h is not None]


@dataclass
class ObjectBatch:
    """Stacked across B Object samples."""
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    verts3d:                 Optional[Tensor] = None  # (B, V, 3)
    verts3d_feats:           Optional[Tensor] = None  # (B, V, F) or (B, V, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, V) or (B, V, V)  bool
    mesh:                    Optional[Mesh]   = None  # shared mesh for all B samples
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[Tensor] = None  # (B,)      int64


@dataclass
class ObjectPairBatch:
    """Stacked across B ObjectPair samples."""
    src_pts3d:                    Optional[Tensor] = None  # (B, N, 3)
    src_pts3d_feats:              Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    src_pts3d_feats_mask:         Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    src_verts3d:                  Optional[Tensor] = None  # (B, V, 3)
    src_verts3d_feats:            Optional[Tensor] = None  # (B, V, F) or (B, V, V, F)
    src_verts3d_feats_mask:       Optional[Tensor] = None  # (B, V) or (B, V, V)  bool
    src_verts3d_part_id:          Optional[Tensor] = None  # (B, V)               int64, -1=unlabeled
    src_mesh:                     Optional[Mesh]   = None  # shared mesh for all B src samples
    src_obj_ncds0c_tform4x4_obj:  Optional[Tensor] = None  # (B, 4, 4)
    src_obj_kpts3d:               Optional[Tensor] = None  # (B, K, 3)
    src_obj_kpts3d_mask:          Optional[Tensor] = None  # (B, K)    bool
    src_category:                 Optional[Tensor] = None  # (B,)      int64
    trgt_pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    trgt_pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    trgt_pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    trgt_verts3d:                 Optional[Tensor] = None  # (B, V, 3)
    trgt_verts3d_feats:           Optional[Tensor] = None  # (B, V, F) or (B, V, V, F)
    trgt_verts3d_feats_mask:      Optional[Tensor] = None  # (B, V) or (B, V, V)  bool
    trgt_verts3d_part_id:         Optional[Tensor] = None  # (B, V)               int64, -1=unlabeled
    trgt_mesh:                    Optional[Mesh]   = None  # shared mesh for all B trgt samples
    src_meshes:                   Optional[list]   = None  # list of B Mesh objects (per-sample)
    trgt_meshes:                  Optional[list]   = None  # list of B Mesh objects (per-sample)
    trgt_obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    trgt_obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    trgt_obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    trgt_category:                Optional[Tensor] = None  # (B,)      int64


def collate_object_pairs(
    samples: list[ObjectPair],
    include: Optional[set[str]] = None,
) -> ObjectPairBatch:
    def _get(attr, side: str):
        vals = [getattr(getattr(s, f"{side}_object"), attr) for s in samples]
        if include and f"{side}_{attr}" not in include:
            return None
        return _stack_field(vals)

    def _get_pad(attr, side: str):
        """Pad-stack variable-length vertex tensors; merge with per-item mask."""
        vals = [getattr(getattr(s, f"{side}_object"), attr) for s in samples]
        if include and f"{side}_{attr}" not in include:
            return None, None
        return _pad_stack_field(vals)

    def _merge_masks(pad_mask, raw_mask):
        if pad_mask is None and raw_mask is None:
            return None
        if pad_mask is None:
            return raw_mask
        if raw_mask is None:
            return pad_mask
        return pad_mask & raw_mask

    def _cat(side: str):
        vals = [getattr(s, f"{side}_object").category_id for s in samples]
        key = f"{side}_category"
        if include and key not in include:
            return None
        return _stack_field([
            torch.tensor(v) if v is not None else None for v in vals
        ])

    src_verts3d,       src_feats_pad_mask   = _get_pad("verts3d",            "src")
    src_verts3d_feats, _src_feats_pad_mask  = _get_pad("verts3d_feats",       "src")
    src_feats_mask_raw, _                   = _get_pad("verts3d_feats_mask",  "src")

    trgt_verts3d,       trgt_feats_pad_mask  = _get_pad("verts3d",            "trgt")
    trgt_verts3d_feats, _trgt_feats_pad_mask = _get_pad("verts3d_feats",       "trgt")
    trgt_feats_mask_raw, _                   = _get_pad("verts3d_feats_mask",  "trgt")

    _src_meshes  = [s.src_object.mesh  for s in samples]
    _trgt_meshes = [s.trgt_object.mesh for s in samples]
    src_meshes_list  = _src_meshes  if any(m is not None for m in _src_meshes)  else None
    trgt_meshes_list = _trgt_meshes if any(m is not None for m in _trgt_meshes) else None

    def _pad_part_ids(raw_list, include_key):
        if not any(p is not None for p in raw_list):
            return None
        if include is not None and include_key not in include:
            return None
        filled = [p if p is not None else torch.full((1,), -1, dtype=torch.int64) for p in raw_list]
        sizes = [p.shape[0] for p in filled]
        V_max = max(sizes)
        B_loc = len(filled)
        out = torch.full((B_loc, V_max), -1, dtype=torch.int64)
        for i, (p, s) in enumerate(zip(filled, sizes)):
            out[i, :s] = p
        return out

    trgt_verts3d_part_id = _pad_part_ids(
        [s.trgt_object.obj_verts_part_id for s in samples], "trgt_obj_verts_part_id"
    )
    src_verts3d_part_id = _pad_part_ids(
        [s.src_object.obj_verts_part_id for s in samples], "src_obj_verts_part_id"
    )

    return ObjectPairBatch(
        src_pts3d                   = _get("pts3d",                   "src"),
        src_pts3d_feats             = _get("pts3d_feats",             "src"),
        src_pts3d_feats_mask        = _get("pts3d_feats_mask",        "src"),
        src_verts3d                 = src_verts3d,
        src_verts3d_feats           = src_verts3d_feats,
        src_verts3d_feats_mask      = _merge_masks(_src_feats_pad_mask, src_feats_mask_raw),
        src_verts3d_part_id         = src_verts3d_part_id,
        src_meshes                  = src_meshes_list,
        src_obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj","src"),
        src_obj_kpts3d              = _get("obj_kpts3d",              "src"),
        src_obj_kpts3d_mask         = _get("obj_kpts3d_mask",         "src"),
        src_category                = _cat("src"),
        trgt_pts3d                   = _get("pts3d",                   "trgt"),
        trgt_pts3d_feats             = _get("pts3d_feats",             "trgt"),
        trgt_pts3d_feats_mask        = _get("pts3d_feats_mask",        "trgt"),
        trgt_verts3d                 = trgt_verts3d,
        trgt_verts3d_feats           = trgt_verts3d_feats,
        trgt_verts3d_feats_mask      = _merge_masks(_trgt_feats_pad_mask, trgt_feats_mask_raw),
        trgt_verts3d_part_id         = trgt_verts3d_part_id,
        trgt_meshes                  = trgt_meshes_list,
        trgt_obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj","trgt"),
        trgt_obj_kpts3d              = _get("obj_kpts3d",              "trgt"),
        trgt_obj_kpts3d_mask         = _get("obj_kpts3d_mask",         "trgt"),
        trgt_category                = _cat("trgt"),
    )


def collate_objects(
    samples: list[Object],
    include: Optional[set[str]] = None,
) -> ObjectBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_pad(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None, None
        return _pad_stack_field(vals)

    def _merge_masks(pad_mask, raw_mask):
        if pad_mask is None and raw_mask is None:
            return None
        if pad_mask is None:
            return raw_mask
        if raw_mask is None:
            return pad_mask
        return pad_mask & raw_mask

    verts3d,       feats_pad_mask   = _get_pad("verts3d")
    verts3d_feats, _feats_pad_mask  = _get_pad("verts3d_feats")
    feats_mask_raw, _               = _get_pad("verts3d_feats_mask")

    return ObjectBatch(
        pts3d                   = _get("pts3d"),
        pts3d_feats             = _get("pts3d_feats"),
        pts3d_feats_mask        = _get("pts3d_feats_mask"),
        verts3d                 = verts3d,
        verts3d_feats           = verts3d_feats,
        verts3d_feats_mask      = _merge_masks(_feats_pad_mask, feats_mask_raw),
        obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj"),
        obj_kpts3d              = _get("obj_kpts3d"),
        obj_kpts3d_mask         = _get("obj_kpts3d_mask"),
        category = _stack_field([
            torch.tensor(s.category_id) if s.category_id is not None else None
            for s in samples
        ]) if (include is None or "category" in include) else None,
    )

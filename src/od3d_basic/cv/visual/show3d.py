from __future__ import annotations
import colorsys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def _mesh_to_trimesh(m):
    import trimesh
    import numpy as np
    vertices = m.verts.numpy()
    faces = m.faces.numpy()
    visual = None
    if m.vert_colors is not None:
        vc = (m.vert_colors.numpy() * 255).clip(0, 255).astype(np.uint8)
        visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
    elif m.texture is not None and m.verts_uvs is not None:
        from PIL import Image
        tex_np = (m.texture.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(tex_np)
        uv = m.verts_uvs.numpy().copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # flip y back to trimesh UV convention
        visual = trimesh.visual.TextureVisuals(uv=uv, image=pil_img)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def _make_kpts_spheres(kpts_np, mask_np, radius: float = 0.02):
    try:
        import trimesh
        import numpy as np
    except ImportError:
        return None

    template = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    n_total = len(kpts_np)
    meshes = []
    for i in range(n_total):
        if not mask_np[i]:
            continue
        r, g, b = colorsys.hsv_to_rgb(i / max(n_total, 1), 0.9, 0.88)
        color = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        s = template.copy()
        s.apply_translation(kpts_np[i])
        s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1))
        meshes.append(s)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def visualize_mesh_dataset(dataset: "Dataset") -> None:
    try:
        import viser
    except ImportError:
        print("\nInstall viser and trimesh: pip install viser trimesh")
        return

    server = viser.ViserServer()
    n = len(dataset)
    idx = [0]
    handles: list = []

    def _clear() -> None:
        for h in handles:
            h.remove()
        handles.clear()

    def _load(i: int) -> None:
        _clear()
        obj = dataset[i]
        oid = obj.object_id

        if obj.modalities.mesh is not None:
            mesh_tm = _mesh_to_trimesh(obj.modalities.mesh)
            h = server.scene.add_mesh_trimesh("/object/mesh", mesh_tm)
            handles.append(h)

        kpts   = obj.modalities.obj_kpts3d
        kpts_m = obj.modalities.obj_kpts3d_mask
        kpts_info = ""
        if kpts is not None and kpts_m is not None:
            import numpy as np
            kpts_np = kpts.numpy()
            mask_np = kpts_m.numpy().astype(bool)
            if mask_np.any():
                kpts_mesh = _make_kpts_spheres(kpts_np, mask_np)
                if kpts_mesh is not None:
                    h = server.scene.add_mesh_trimesh("/object/kpts3d", kpts_mesh)
                    handles.append(h)
            kpts_info = f"  kpts={mask_np.sum()}/{len(mask_np)}"

        obj_label.value = f"[{i + 1}/{n}]  {oid}"
        print(f"  [{i + 1}/{n}] {oid}{kpts_info}")

    with server.gui.add_folder("Navigation"):
        obj_label = server.gui.add_text("Object", initial_value="loading…")
        btn_prev  = server.gui.add_button("← Prev")
        btn_next  = server.gui.add_button("Next →")

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
    print("Use Prev / Next in the panel to browse objects. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")

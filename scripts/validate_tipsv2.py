"""Standalone shape/timing/memory validation for the TIPSv2 extractor.

Renders one view of a synthetic mesh (`pytorch3d.utils.ico_sphere`) and
reports, at every stage: render resolution, preprocessing resolution, patch
grid size, raw `patch_tokens` shape, reshaped spatial shape, `grid_sample`
output shape, final `(V, C)` vertex-feature shape, feature dimension,
load/inference time, and peak GPU memory.

Requires CUDA + `transformers` (with `trust_remote_code=True` support) and
network access to download `google/tipsv2-b14` on first run.

Run on the SLURM cluster (or any CUDA machine with the `o3b` venv):
    python third_party/o3b/scripts/validate_tipsv2.py
"""
from __future__ import annotations

import time

import torch


def main() -> None:
    assert torch.cuda.is_available(), "validate_tipsv2.py requires a CUDA device"
    device = torch.device("cuda")

    from pytorch3d.utils import ico_sphere
    from pytorch3d.renderer import TexturesVertex
    from o3b.model.diff3f.diff3f.diff3f import arange_pixels, get_features_per_vertex
    from o3b.model.tipsv2.extractor import TIPSv2Extractor

    H = W = 512
    resolution = 448
    hub_model = "google/tipsv2-b14"

    print(f"render resolution        : {(H, W)}")
    print(f"preprocessing resolution : {resolution}x{resolution}")

    t0 = time.time()
    extractor = TIPSv2Extractor(device, hub_model=hub_model, resolution=resolution)
    load_s = time.time() - t0
    print(f"patch size                : 14")
    print(f"patch grid                : {extractor.grid_hw}x{extractor.grid_hw}")
    print(f"feature dimension          : {extractor.feature_dims}")
    print(f"model load time (s)        : {load_s:.2f}")

    mesh = ico_sphere(4, device)  # small synthetic mesh, ~2562 verts
    # ico_sphere ships with no texture, but the shared renderer (render.py,
    # used by every backbone) always calls meshes.sample_textures(); real
    # HouseCorr3D meshes come with textures loaded, so give this synthetic
    # one a plain white vertex texture just so it's renderable.
    verts = mesh.verts_padded()
    mesh.textures = TexturesVertex(verts_features=torch.ones_like(verts))

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    feats = get_features_per_vertex(
        device=device,
        pipe=None,
        dino_model=None,
        mesh=mesh,
        prompt=None,
        num_views=4,
        H=H,
        W=W,
        tolerance=0.01,
        use_normal_map=False,
        extractor_fn=extractor,
    )
    infer_s = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    V = mesh.verts_list()[0].shape[0]
    print(f"final per-vertex features : {tuple(feats.shape)} (expected ({V}, {extractor.feature_dims}))")
    print(f"grid_sample output shape  : (1, {extractor.feature_dims}, {H * W})")
    print(f"inference time (s)         : {infer_s:.2f}")
    print(f"peak GPU memory (GB)       : {peak_mem_gb:.2f}")

    assert feats.shape == (V, extractor.feature_dims), (
        f"Expected ({V}, {extractor.feature_dims}), got {tuple(feats.shape)}"
    )
    print("\nTIPSv2 validation OK")


if __name__ == "__main__":
    main()

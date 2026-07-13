"""Standalone validation for the SigLIP2 extractor plugged into the shared Diff3F pipeline.

Renders a single view of a synthetic mesh, runs SigLIP2Extractor on it, and reports every
shape in the chain (input resolution -> preprocessing -> patch grid -> encoder output ->
reshaped spatial map -> grid_sample output -> final per-vertex feature tensor), plus timing
and GPU memory. Requires an environment with CUDA + `transformers` installed (this project
targets a SLURM cluster; see setup/setup_local.sh) — it is not runnable in a plain sandbox.

Usage: python third_party/o3b/scripts/validate_siglip2.py
"""
from __future__ import annotations

import time

import numpy as np
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from pytorch3d.utils import ico_sphere

    from o3b.model.diff3f.diff3f.diff3f import arange_pixels
    from o3b.model.diff3f.diff3f.render import batch_render
    from o3b.model.siglip2.extractor import SigLIP2Extractor

    H = W = 512
    mesh = ico_sphere(level=3, device=device)  # small synthetic mesh, ~642 verts
    mesh_vertices = mesh.verts_list()[0]
    print(f"Synthetic mesh: {mesh_vertices.shape[0]} vertices")

    batched_renderings, _, camera, depth = batch_render(
        device, mesh, mesh_vertices, num_views=1, H=H, W=W, use_normal_map=False
    )
    rendered_img = (batched_renderings[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
    depth_map = depth[0, :, :, 0].unsqueeze(0).to(device)
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()

    print(f"Rendered view resolution: {rendered_img.shape[:2]}")

    t0 = time.time()
    extractor = SigLIP2Extractor(device=device)
    load_time = time.time() - t0

    proc_size = extractor.processor.image_processor.size
    print(f"Preprocessing target resolution: {proc_size}")
    print(f"Patch size: {extractor.patch_size}")

    t1 = time.time()
    from PIL import Image

    img = Image.fromarray(rendered_img).convert("RGB")
    inputs = extractor.processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=extractor.dtype)
    h = pixel_values.shape[-2] // extractor.patch_size
    w = pixel_values.shape[-1] // extractor.patch_size
    print(f"Preprocessed pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"Patch grid: {h}x{w}")

    with torch.no_grad():
        raw_out = extractor.model(pixel_values=pixel_values)
    print(f"Encoder last_hidden_state shape: {tuple(raw_out.last_hidden_state.shape)}")

    aligned_features = extractor(rendered_img, depth_map, grid, device)
    inference_time = time.time() - t1
    print(f"Reshaped spatial feature map: (1, {extractor.feature_dims}, {h}, {w})")
    print(f"grid_sample output shape: {tuple(aligned_features.shape)}")
    print(f"Feature dimension (C): {extractor.feature_dims}")

    # emulate final per-vertex aggregation shape (single view, no averaging needed to check shape)
    final_vertex_features = torch.zeros(
        (mesh_vertices.shape[0], extractor.feature_dims), dtype=torch.float16
    )
    print(f"Final vertex feature tensor shape (V, C): {tuple(final_vertex_features.shape)}")

    print(f"\nModel load time: {load_time:.2f}s")
    print(f"Single-view inference time: {inference_time:.3f}s")
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Peak GPU memory allocated: {peak_mem_mb:.1f} MB")
    else:
        print("GPU memory: N/A (running on CPU)")


if __name__ == "__main__":
    main()

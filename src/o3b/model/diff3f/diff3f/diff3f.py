import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from dataclasses import dataclass
from o3b.model.diff3f.diff3f.extractors import Diff3FExtractor
from o3b.model.diff3f.diff3f.render import batch_render
from tqdm import tqdm
from time import time
import random

#FEATURE_DIMS = 768 # dino
FEATURE_DIMS = 2048 #  1280+768 # diffusion unet + dino

VERTEX_GPU_LIMIT = 35000

AGGREGATION_MODES = ("mean", "all_views")


def _log_rss(tag: str) -> None:
    """Temporary diagnostic: print peak host RSS so far (high-water mark, Linux).
    Used to pin down exactly which step in get_features_per_vertex's all_views
    path is responsible for OOM-killed DataLoader workers on Colab — remove once
    the culprit is confirmed."""
    import resource
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[rss] {tag}: peak RSS so far = {peak_mb:.0f} MB", flush=True)


@dataclass
class PerVertexFeatures:
    """Per-vertex features returned by get_features_per_vertex(aggregation_mode="all_views").

    mean:           (V, F) half — identical to the aggregation_mode="mean" return value.
    all_views:      (V, K_max, F) half — one averaged feature observation per vertex per
                     rendered view (in-view duplicate pixel hits on the same vertex are
                     averaged together first — see note in get_features_per_vertex),
                     zero-padded so K_max <= num_views.
    all_views_mask: (V, K_max) bool — True where all_views holds a real observation.
    """
    mean: torch.Tensor
    all_views: torch.Tensor
    all_views_mask: torch.Tensor


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    device,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
    extractor_fn=None,
    aggregation_mode="mean",
):
    if aggregation_mode not in AGGREGATION_MODES:
        raise ValueError(
            f"aggregation_mode must be one of {AGGREGATION_MODES}, got {aggregation_mode!r}"
        )
    t1 = time()
    extractor = extractor_fn or Diff3FExtractor(
        pipe,
        dino_model,
        prompt,
        use_latent=use_latent,
        num_images_per_prompt=num_images_per_prompt,
        return_image=return_image,
        prompts_list=prompts_list,
    )
    feature_dims = getattr(extractor, "feature_dims", FEATURE_DIMS)
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    _log_rss("before rendering")
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map
    )
    print("Rendering complete")
    _log_rss("after rendering")
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), feature_dims)).half()  # .to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)
    # aggregation_mode="all_views": in parallel with the running sum/count above,
    # keep one feature observation per vertex per rendered view. A single view can
    # assign multiple pixels to the same vertex (ball_query returns up to K=100
    # neighbours per pixel, and even the nearest-vertex fallback can have several
    # pixels share a nearest vertex), so those in-view duplicates are averaged
    # together via index_add_ (correct scatter-add, unlike ft_per_vertex's `+=`
    # above) before being kept — this bounds K_max by num_views instead of by
    # total raw pixel-vertex hits, which otherwise scales with render resolution
    # and blows up memory (V, K_max, F) for coarse meshes at high resolution.
    keep_all_views = aggregation_mode == "all_views"
    all_views_vidx_chunks = [] if keep_all_views else None
    all_views_feat_chunks = [] if keep_all_views else None
    V_total = len(mesh_vertices)

    def _dedupe_view_obs(vidx: torch.Tensor, feat: torch.Tensor):
        """Collapse possibly-duplicate (vertex_idx, feature) pairs from one
        rendered view into at most one averaged observation per vertex."""
        sums = torch.zeros((V_total, feature_dims), dtype=torch.half)
        counts = torch.zeros((V_total, 1), dtype=torch.half)
        sums.index_add_(0, vidx, feat)
        counts.index_add_(0, vidx, torch.ones((vidx.shape[0], 1), dtype=torch.half))
        has_obs = counts[:, 0] != 0
        view_vidx = torch.nonzero(has_obs, as_tuple=True)[0]
        view_feat = sums[view_vidx] / counts[view_vidx]
        return view_vidx, view_feat
    for idx in tqdm(range(len(batched_renderings))):
        if idx % 20 == 0:
            _log_rss(f"view {idx}")
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device)
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)

        aligned_features = extractor(
            diffusion_input_img, depth_map, grid, device, normal_map_input=normal_map_input
        )

        features_per_pixel = aligned_features[0, :, indices].cpu()
        # map pixel to vertex on mesh
        if bq:
            from pytorch3d.ops import ball_query
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            vidx_obs = queried_indices[mask]
            feat_obs = features_per_pixel.repeat_interleave(repeat, dim=1).T
            ft_per_vertex_count[vidx_obs] += 1
            ft_per_vertex[vidx_obs] += feat_obs
            if keep_all_views:
                view_vidx, view_feat = _dedupe_view_obs(vidx_obs, feat_obs)
                all_views_vidx_chunks.append(view_vidx)
                all_views_feat_chunks.append(view_feat)
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1
            if keep_all_views:
                view_vidx, view_feat = _dedupe_view_obs(
                    closest_vertex_indices, features_per_pixel.T
                )
                all_views_vidx_chunks.append(view_vidx)
                all_views_feat_chunks.append(view_feat)

    _log_rss("after per-view loop")
    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print("Number of missing features: ", missing_features)
    print("Copied features from nearest vertices")

    all_views = all_views_mask = None
    if keep_all_views:
        V = len(mesh_vertices)
        if all_views_vidx_chunks:
            n_chunks = len(all_views_vidx_chunks)
            print(f"[rss] all_views: {n_chunks} view-chunks to concat", flush=True)
            vidx_cat = torch.cat(all_views_vidx_chunks)             # (Total,) long
            feat_cat = torch.cat(all_views_feat_chunks, dim=0)      # (Total, F) half
            print(f"[rss] all_views: vidx_cat={tuple(vidx_cat.shape)} "
                  f"feat_cat={tuple(feat_cat.shape)} dtype={feat_cat.dtype}", flush=True)
            _log_rss("after concat")

            order = torch.argsort(vidx_cat, stable=True)
            vidx_sorted = vidx_cat[order]
            feat_sorted = feat_cat[order]
            _log_rss("after sort")

            obs_counts = torch.bincount(vidx_sorted, minlength=V)
            K_max = int(obs_counts.max().item())
            print(f"[rss] all_views: V={V} K_max={K_max} "
                  f"(final tensor {V}x{K_max}x{feature_dims} half "
                  f"= {V*K_max*feature_dims*2/1e6:.0f} MB)", flush=True)

            # position of each observation within its vertex's contiguous run,
            # vectorized (no per-observation Python loop):
            change = torch.ones_like(vidx_sorted, dtype=torch.bool)
            change[1:] = vidx_sorted[1:] != vidx_sorted[:-1]
            group_start = torch.nonzero(change, as_tuple=True)[0]
            group_id = torch.cumsum(change.long(), dim=0) - 1
            slot = torch.arange(vidx_sorted.shape[0]) - group_start[group_id]

            all_views = torch.zeros((V, K_max, feature_dims), dtype=torch.half)
            all_views_mask = torch.zeros((V, K_max), dtype=torch.bool)
            all_views[vidx_sorted, slot] = feat_sorted
            all_views_mask[vidx_sorted, slot] = True
            _log_rss("after all_views tensor built")
        else:
            all_views = torch.zeros((V, 0, feature_dims), dtype=torch.half)
            all_views_mask = torch.zeros((V, 0), dtype=torch.bool)

    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][
            closest_vertex_indices, :
        ]
        if keep_all_views:
            # mirror the same nearest-vertex copy for the raw observations, so a
            # vertex with zero direct observations still carries the borrowed
            # vertex's full set of per-view features (mask included) rather than
            # an all-False row.
            all_views[missing_indices, :, :] = all_views[filled_indices][
                closest_vertex_indices, :, :
            ]
            all_views_mask[missing_indices, :] = all_views_mask[filled_indices][
                closest_vertex_indices, :
            ]
    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    if aggregation_mode == "all_views":
        return PerVertexFeatures(
            mean=ft_per_vertex, all_views=all_views, all_views_mask=all_views_mask,
        )
    return ft_per_vertex

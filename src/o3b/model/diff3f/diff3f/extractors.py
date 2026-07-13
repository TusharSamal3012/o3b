"""Encoder-specific feature extractors pluggable into get_features_per_vertex.

An extractor is any callable with the signature:

    extractor(rendered_img: np.ndarray, depth_map: torch.Tensor, grid: torch.Tensor,
              device, normal_map_input=None) -> torch.Tensor  # (1, C, H*W) fp16, normalized over C

and an int `feature_dims` attribute. get_features_per_vertex owns rendering, unprojection,
ball_query vertex assignment, multi-view averaging, and missing-vertex fill; only the pixel
feature extraction (this file) differs between backbones.
"""
import random

import torch

from o3b.model.diff3f.diff3f.diffusion import add_texture_to_render
from o3b.model.diff3f.diff3f.dino import get_dino_features

FEATURE_DIMS = 2048  # 1280 (SD UNet) + 768 (DINOv2-B)


class Diff3FExtractor:
    """Default Diff3F encoder: SD ControlNet UNet features fused 50/50 with DINOv2-B."""

    feature_dims = FEATURE_DIMS

    def __init__(
        self,
        pipe,
        dino_model,
        prompt,
        use_latent=False,
        num_images_per_prompt=1,
        return_image=True,
        prompts_list=None,
    ):
        self.pipe = pipe
        self.dino_model = dino_model
        self.prompt = prompt
        self.use_latent = use_latent
        self.num_images_per_prompt = num_images_per_prompt
        self.return_image = return_image
        self.prompts_list = prompts_list

    def __call__(self, rendered_img, depth_map, grid, device, normal_map_input=None):
        prompt = self.prompt
        if self.prompts_list is not None:
            prompt = random.choice(self.prompts_list)

        diffusion_output = add_texture_to_render(
            self.pipe,
            rendered_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=self.use_latent,
            num_images_per_prompt=self.num_images_per_prompt,
            return_image=self.return_image,
        )

        aligned_dino_features = get_dino_features(
            device, self.dino_model, diffusion_output[1][0], grid
        )
        with torch.no_grad():
            ft = torch.nn.Upsample(size=grid.shape[1:3], mode="bilinear")(
                diffusion_output[0].unsqueeze(0)
            ).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)

        return torch.hstack([aligned_features * 0.5, aligned_dino_features * 0.5])

"""DINOv3 dense feature extractor for the shared Diff3F projection pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


class DINOv3Extractor:
    def __init__(
        self,
        hub_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        device=None,
        dtype=torch.float16,
        image_size=384,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.image_size = int(image_size)

        self.processor = AutoImageProcessor.from_pretrained(hub_model)
        self.model = (
            AutoModel.from_pretrained(hub_model, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

        self.patch_size = int(self.model.config.patch_size)
        self.feature_dims = int(self.model.config.hidden_size)
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0)
        )

        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}"
            )

        self.last_input_shape = None
        self.last_token_shape = None
        self.last_patch_grid = None
        self.last_prefix_tokens = None

    @torch.inference_mode()
    def __call__(
        self,
        rendered_img,
        depth_map,
        grid,
        device,
        normal_map_input=None,
    ):
        image = Image.fromarray(rendered_img).convert("RGB")
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.dtype,
        )

        height, width = pixel_values.shape[-2:]
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"DINOv3 processor produced {(height, width)}, expected "
                f"{(self.image_size, self.image_size)}"
            )

        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        expected_patches = patch_h * patch_w

        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state

        prefix_tokens = tokens.shape[1] - expected_patches
        expected_prefix = 1 + self.num_register_tokens

        if prefix_tokens != expected_prefix:
            raise ValueError(
                f"Expected {expected_prefix} CLS/register tokens and "
                f"{expected_patches} patch tokens, but received "
                f"{tokens.shape[1]} total tokens"
            )

        # DINOv3 token order: CLS, register tokens, then spatial patches.
        features = tokens[:, -expected_patches:, :]
        features = features.reshape(
            features.shape[0],
            patch_h,
            patch_w,
            self.feature_dims,
        ).permute(0, 3, 1, 2)

        features = F.grid_sample(
            features,
            grid.to(device=features.device, dtype=features.dtype),
            align_corners=False,
        )
        features = features.reshape(
            features.shape[0],
            self.feature_dims,
            -1,
        )
        features = F.normalize(features, dim=1)

        self.last_input_shape = tuple(pixel_values.shape)
        self.last_token_shape = tuple(tokens.shape)
        self.last_patch_grid = (patch_h, patch_w)
        self.last_prefix_tokens = prefix_tokens

        return features

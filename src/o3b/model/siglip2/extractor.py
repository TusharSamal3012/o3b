"""SigLIP2 encoder as a pluggable Diff3F extractor.

Implements the extractor interface consumed by get_features_per_vertex
(o3b.model.diff3f.diff3f.diff3f): a callable that turns one rendered view into a
(1, C, H*W) normalized dense feature map. Rendering, unprojection, ball_query vertex
assignment, multi-view averaging, and missing-vertex fill are all owned by
get_features_per_vertex and are not duplicated here.
"""
import torch
import torch.nn.functional as F
from PIL import Image


class SigLIP2Extractor:
    def __init__(self, hub_model="google/siglip2-base-patch16-384", device=None, dtype=torch.float16):
        from transformers import AutoConfig, AutoProcessor, SiglipVisionModel, Siglip2VisionModel

        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(hub_model)

        # Fixed-resolution SigLIP2 checkpoints (e.g. the "-384" release) were trained with
        # the SigLIP2 recipe but kept the original SigLIP v1 architecture (Conv2d patch
        # embedding, fixed position embeddings) — their vision_config.model_type is still
        # "siglip_vision_model", not "siglip2_vision_model". Only "-naflex" checkpoints use
        # the newer NaFlex architecture that Siglip2VisionModel expects. Loading the wrong
        # class raises a shape-mismatch error, so pick the class from the checkpoint's own
        # declared type instead of assuming.
        vision_model_type = getattr(
            getattr(AutoConfig.from_pretrained(hub_model), "vision_config", None),
            "model_type", "siglip2_vision_model",
        )
        model_cls = SiglipVisionModel if vision_model_type == "siglip_vision_model" else Siglip2VisionModel
        self.model = (
            model_cls.from_pretrained(hub_model, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        self.patch_size = self.model.config.patch_size
        self.feature_dims = self.model.config.hidden_size

        # google/siglip2-*-patch16-384 is a fixed-resolution checkpoint: the projection stage
        # (grid_sample in diff3f.py) assumes a clean square h == w patch grid derived from the
        # preprocessed image size. Force resize-only, no-crop preprocessing explicitly rather
        # than trusting whatever the installed transformers version defaults to.
        image_size = getattr(self.model.config, "image_size", 384)
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.do_center_crop = False
        self.processor.image_processor.size = {"height": image_size, "width": image_size}

    @torch.no_grad()
    def __call__(self, rendered_img, depth_map, grid, device, normal_map_input=None):
        img = Image.fromarray(rendered_img).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=self.dtype)

        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size
        if pixel_values.shape[-2] % self.patch_size or pixel_values.shape[-1] % self.patch_size:
            raise ValueError(
                f"SigLIP2 preprocessing produced a {tuple(pixel_values.shape[-2:])} image not "
                f"divisible by patch size {self.patch_size}; patch grid would be ambiguous."
            )

        out = self.model(pixel_values=pixel_values)
        features = out.last_hidden_state  # (1, h*w, C), no CLS token
        if features.shape[1] != h * w:
            raise ValueError(
                f"Expected {h * w} patch tokens for a {h}x{w} grid, got {features.shape[1]} — "
                "SigLIP2 preprocessing does not match the assumed fixed-resolution behavior."
            )

        features = features.reshape(1, h, w, self.feature_dims).permute(0, 3, 1, 2)
        features = F.grid_sample(features, grid.to(self.dtype), align_corners=False)
        features = features.reshape(1, self.feature_dims, -1)
        features = F.normalize(features, dim=1)
        return features

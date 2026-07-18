"""TIPSv2 2D-feature extractor.

TIPSv2 (google-deepmind/tips, HuggingFace `google/tipsv2-*`) is a ViT
image-text contrastive encoder trained with an explicit spatial-awareness
objective (masked image modeling + self-distillation on top of contrastive
image-text learning). It is loaded via HF `AutoModel` with
`trust_remote_code=True` (custom modeling code, not a stock `transformers`
architecture like SigLIP2).

Like SigLIP2, TIPSv2 has no CLS token mixed into its patch sequence: the
model's own `encode_image()` call already returns the global embedding
(`out.cls_token`, shape `(1, 1, C)`) and the dense patch grid
(`out.patch_tokens`, shape `(1, N, C)`) as two separate fields, so there is
nothing to strip before reshaping to a spatial map.

Implements the same interface as `Diff3FExtractor`
(o3b/model/diff3f/diff3f/extractors.py) and `SigLIP2Extractor`:

    extractor(rendered_img: np.ndarray,   # (H, W, 3) uint8, the rendered view
              depth_map:    torch.Tensor, # (1, H, W) — unused by TIPSv2
              grid:         torch.Tensor, # (1, H, W, 2), pixel sampling grid
              device:       torch.device,
              normal_map_input=None,      # unused by TIPSv2
             ) -> torch.Tensor            # (1, C, H*W) normalised, fp16
    extractor.feature_dims: int
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Embed dim per published variant (see the model card table on
# https://huggingface.co/google/tipsv2-b14).
TIPSV2_FEATURE_DIMS = {
    "google/tipsv2-b14": 768,
    "google/tipsv2-l14": 1024,
    "google/tipsv2-so400m14": 1152,
    "google/tipsv2-g14": 1536,
}

_PATCH_SIZE = 14  # all TIPSv2 variants use a 14x14 patch


class TIPSv2Extractor:
    """Loads TIPSv2 once and extracts a dense (1, C, H*W) feature map per
    rendered view, reusing the identical grid_sample + normalize tail that
    DINOv2/SigLIP2 already use inside `get_features_per_vertex`.
    """

    def __init__(
        self,
        device: torch.device,
        hub_model: str = "google/tipsv2-b14",
        resolution: int = 448,
    ):
        from transformers import AutoModel

        if resolution % _PATCH_SIZE != 0:
            raise ValueError(
                f"TIPSv2 resolution must be a multiple of the {_PATCH_SIZE}px "
                f"patch size, got {resolution}."
            )
        self.hub_model = hub_model
        self.resolution = resolution
        self.grid_hw = resolution // _PATCH_SIZE
        self.feature_dims = TIPSV2_FEATURE_DIMS.get(hub_model, 768)
        self.device = device

        logger.info("TIPSv2Extractor: loading %s (trust_remote_code)…", hub_model)
        model = AutoModel.from_pretrained(hub_model, trust_remote_code=True)
        self.model = model.to(device).eval().half()

    @torch.no_grad()
    def __call__(self, rendered_img, depth_map, grid, device, normal_map_input=None):
        from torchvision import transforms as tfs

        # TIPSv2 expects [0, 1] tensors with NO ImageNet/CLIP normalisation,
        # resized to a multiple of the 14px patch size, no center-crop.
        transform = tfs.Compose(
            [
                tfs.ToPILImage(),
                tfs.Resize((self.resolution, self.resolution)),
                tfs.ToTensor(),
            ]
        )
        img = transform(rendered_img).unsqueeze(0).to(device).half()  # (1, 3, res, res)

        out = self.model.encode_image(img)
        patch_tokens = out.patch_tokens  # (1, grid_hw*grid_hw, C) — no CLS token to strip
        h = w = self.grid_hw
        dim = patch_tokens.shape[-1]
        assert patch_tokens.shape[1] == h * w, (
            f"Expected {h * w} patch tokens for a {self.resolution}px / "
            f"{_PATCH_SIZE}px grid, got {patch_tokens.shape[1]}."
        )

        features = patch_tokens.reshape(1, h, w, dim).permute(0, 3, 1, 2).half()  # (1, C, h, w)
        features = F.grid_sample(features, grid, align_corners=False).reshape(1, dim, -1)
        features = F.normalize(features, dim=1)
        return features

from __future__ import annotations

import logging
from functools import reduce
from operator import mul
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


def _get_block(
    block_type: str,
    in_dim: int,
    out_dim: int,
    stride: int,
    out_conv1x1: bool = False,
    pre_upsampling: float = 1.0,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    if pre_upsampling != 1.0:
        modules.append(nn.Upsample(scale_factor=pre_upsampling))

    downsample = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_dim),
    )
    if block_type == "bottleneck":
        modules.append(Bottleneck(inplanes=in_dim, planes=out_dim // 4, stride=stride, downsample=downsample))
    elif block_type == "basic":
        modules.append(BasicBlock(inplanes=in_dim, planes=out_dim, stride=stride, downsample=downsample))
    else:
        raise ValueError(f"Unknown ResNet block type: {block_type!r}")

    if out_conv1x1:
        modules.append(nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, bias=True))

    return nn.Sequential(*modules)


@register_model("ResNetHead")
class ResNetHead(OD3D_Model):
    """ResNet-based feature-map refinement head for o3b.

    Mirrors od3d.models.heads.resnet.head.ResNet for the single-scale case
    (no FPN).  Input: frames_pred.featmap (B, in_dim, H, W).
    Output: frames_pred.featmap (B, out_dim, H', W').

    NeMo checkpoints trained with od3d can be loaded via load_state_dict
    (after stripping the 'head.' prefix) with strict=False to skip the
    od3d-specific fc layer (which processes the CLS token; unused here).
    """

    def __init__(
        self,
        in_dim: int,
        block_type: str = "bottleneck",
        conv_blocks: Optional[dict] = None,
        fully_connected: Optional[dict] = None,
        normalize: bool = True,
        pad_zero: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.pad_zero = pad_zero
        self.pad_width = 1

        conv_blocks    = conv_blocks    or {}
        fully_connected = fully_connected or {}

        out_dims       = list(conv_blocks.get("out_dims",       []))
        strides        = list(conv_blocks.get("strides",        [1]   * len(out_dims)))
        out_conv1x1    = list(conv_blocks.get("out_conv1x1",    [False] * len(out_dims)))
        pre_upsampling = list(conv_blocks.get("pre_upsampling", [1.0] * len(out_dims)))

        n = len(out_dims)
        assert len(strides) == n and len(out_conv1x1) == n and len(pre_upsampling) == n

        in_dims_seq = [in_dim] + out_dims[:-1]

        conv_block_scaling = [strides[i] / pre_upsampling[i] for i in range(n)]
        self.downsample_rate = reduce(mul, [1.0] + conv_block_scaling, 1.0)

        self.conv_blocks = nn.Sequential(*[
            _get_block(
                block_type=block_type,
                in_dim=in_dims_seq[i],
                out_dim=out_dims[i],
                stride=strides[i],
                out_conv1x1=out_conv1x1[i],
                pre_upsampling=pre_upsampling[i],
            )
            for i in range(n)
        ])

        fc_out_dim = fully_connected.get("out_dim", None)
        if fc_out_dim is not None:
            self.fc_enabled = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(out_dims[-1] if out_dims else in_dim, fc_out_dim)
            self.out_dim = fc_out_dim
        else:
            self.fc_enabled = False
            self.out_dim = out_dims[-1] if out_dims else in_dim

    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt

        x = frames_pred.featmap  # (B, in_dim, H, W)

        if self.pad_zero:
            x = F.pad(x, (self.pad_width,) * 4, mode="constant", value=0)

        x = self.conv_blocks(x)

        if self.pad_zero:
            pw = int((1.0 / self.downsample_rate) * self.pad_width)
            x = x[:, :, pw:-pw, pw:-pw]

        if self.fc_enabled:
            feat = torch.flatten(self.avgpool(x), 1)
            feat = self.fc(feat)
            frames_pred.feat = feat

        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        frames_pred.featmap = x
        return frames_gt, frames_pred

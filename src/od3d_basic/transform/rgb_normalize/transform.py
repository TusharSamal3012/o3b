from typing import List, Union

import torch
from torchvision.transforms.transforms import Normalize

from od3d_basic.data.datatypes.frame import Frame, FrameBatch
from od3d_basic.transform.transform import FrameTransform


class RGB_Normalize(FrameTransform):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, data: Union[Frame, FrameBatch]) -> Union[Frame, FrameBatch]:
        if data.rgb is not None:
            if data.rgb.dim() == 4:  # FrameBatch: (B, 3, H, W)
                data.rgb = torch.stack([self.normalize(rgb) for rgb in data.rgb], dim=0)
            else:  # Frame: (3, H, W)
                data.rgb = self.normalize(data.rgb)
        return data

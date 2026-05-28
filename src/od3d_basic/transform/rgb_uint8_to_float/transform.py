from typing import Union

from od3d_basic.data.datatypes.frame import Frame, FrameBatch
from od3d_basic.transform.transform import FrameTransform


class RGB_UInt8ToFloat(FrameTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Union[Frame, FrameBatch]) -> Union[Frame, FrameBatch]:
        if data.rgb is not None:
            data.rgb = data.rgb.float() / 255.0
        return data

import logging
from typing import List, Union

from omegaconf import DictConfig

from o3b.data.datatypes.frame import Frame, FrameBatch
from o3b.transform.transform import FrameTransform

logger = logging.getLogger(__name__)


class SequentialTransform(FrameTransform):
    def __init__(self, transforms: List[FrameTransform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data: Union[Frame, FrameBatch]) -> Union[Frame, FrameBatch]:
        for transform in self.transforms:
            if transform is not None:
                data = transform(data)
            else:
                logger.warning(f"Transform is None. {self.transforms}")
        return data

    @classmethod
    def create_from_config(cls, config: DictConfig) -> "SequentialTransform":
        transforms: List[FrameTransform] = []
        for config_transform in config.transforms:
            transforms.append(
                FrameTransform.subclasses[config_transform.class_name].create_from_config(
                    config=config_transform,
                ),
            )
        return SequentialTransform(transforms)

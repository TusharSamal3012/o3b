import importlib
import inspect
from typing import Union

from omegaconf import DictConfig

from o3b.data.datatypes.frame import Frame, FrameBatch


_CLASS_TO_MODULE: dict[str, str] = {
    "SequentialTransform": "o3b.transform.sequential.transform",
    "RGB_UInt8ToFloat":    "o3b.transform.rgb_uint8_to_float.transform",
    "RGB_Normalize":       "o3b.transform.rgb_normalize.transform",
}


def _ensure_transform_imported(name: str) -> None:
    if name not in FrameTransform.subclasses and name in _CLASS_TO_MODULE:
        importlib.import_module(_CLASS_TO_MODULE[name])


class FrameTransform:
    subclasses: dict[str, type["FrameTransform"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Union[Frame, FrameBatch]) -> Union[Frame, FrameBatch]:
        pass

    @classmethod
    def create_from_config(cls, config: DictConfig) -> "FrameTransform":
        name = config.class_name
        _ensure_transform_imported(name)
        if name not in cls.subclasses:
            raise KeyError(
                f"Unknown transform '{name}'. Registered: {sorted(cls.subclasses)}"
            )
        keys = inspect.getfullargspec(cls.subclasses[name].__init__)[0][1:]
        return cls.subclasses[name](
            **{k: config.get(k) for k in keys if config.get(k, None) is not None},
        )

import inspect
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig


class O3B_Transform:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def get_as_dict(self):
        _dict = {}
        keys = inspect.getfullargspec(self.__init__)[0][1:]
        for key in keys:
            if hasattr(self, key):
                _dict[key] = getattr(self, key)
                if isinstance(_dict[key], torch.Tensor):
                    _dict[key] = _dict[key].detach().cpu().tolist()
                if isinstance(_dict[key], np.ndarray):
                    _dict[key] = _dict[key].tolist()
                if (
                    isinstance(_dict[key], List)
                    and len(_dict[key]) > 0
                    and isinstance(_dict[key][0], O3B_Transform)
                ):
                    _dict[key] = [t.get_as_dict() for t in _dict[key]]
                if isinstance(_dict[key], Enum):
                    _dict[key] = str(_dict[key])
        _dict["class_name"] = type(self).__name__
        return _dict

    def __init__(self, **kwargs):
        pass

    def __call__(self, item):
        pass

    @classmethod
    def create_from_config(cls, config: DictConfig):
        keys = inspect.getfullargspec(cls.subclasses[config.class_name].__init__)[0][1:]
        return cls.subclasses[config.class_name](
            **{
                key: config.get(key)
                for key in keys
                if config.get(key, None) is not None
            },
        )

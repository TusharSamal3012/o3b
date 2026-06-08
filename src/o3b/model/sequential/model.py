import logging

logger = logging.getLogger(__name__)
from typing import List
from omegaconf import DictConfig
from o3b.model.model import OD3D_Model, register_model
import torch

@register_model("SequentialModel")
class SequentialModel(OD3D_Model):
    def __init__(self, models: torch.nn.ModuleList):
        super().__init__()

        self.models: torch.nn.ModuleList = models

    def forward(self, frames_gt, frames_pred=None):
        for model in self.models:
            if model is not None:
                frames_gt, frames_pred = model.forward(frames_gt=frames_gt, frames_pred=frames_pred)
            else:
                logger.warning(f"Model is None. {self.models}")
        return frames_gt, frames_pred

    def get_down_sample_rate(self):
        down_sample_rate = 1.
        for model in self.models:
            if model is not None:
                down_sample_rate *= model.get_down_sample_rate()
            else:
                logger.warning(f"Model is None. {self.models}")
        return down_sample_rate
    

    @classmethod
    def create_from_config(cls, config: DictConfig):
        models: List[OD3D_Model] = []
        for config_transform in config.models:
            models.append(
                OD3D_Model.subclasses[
                    config_transform.class_name
                ].create_from_config(config=config_transform),
            )
        
        models = torch.nn.ModuleList(models)

        return SequentialModel(models)

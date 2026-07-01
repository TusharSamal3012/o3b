import logging

logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from o3b.model.model import OD3D_Model, register_model
import torch


@register_model("SequentialModel")
class SequentialModel(OD3D_Model):
    def __init__(self, models: torch.nn.ModuleList):
        super().__init__()
        self.models: torch.nn.ModuleList = models

    @property
    def out_dim(self):
        for m in reversed(self.models):
            d = getattr(m, "out_dim", None)
            if d is not None:
                return d
        return None

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
        from o3b.model.model import build_model
        models = torch.nn.ModuleList([build_model(m) for m in config.models])
        return SequentialModel(models)

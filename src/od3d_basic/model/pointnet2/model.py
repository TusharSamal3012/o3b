import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from od3d_basic.model.model import OD3D_Model, register_model
import torch
try:
    from pointnet2.pointnet2_modules import Pointnet2ClsMSGFus
except ImportError as e:
    logger.warning(f"pointnet2 not available, skipping registration: {e}")
    Pointnet2ClsMSGFus = None
from od3d_basic.data.datatypes.frame import FrameBatch

@register_model("PointNet2")
class PointNet2(OD3D_Model):
    def __init__(
        self,
        in_dim,
    ):
        super().__init__()
        self.net = Pointnet2ClsMSGFus(input_channels=in_dim)
        
    def forward(self, frames_gt: FrameBatch, frames_pred: FrameBatch = None):
        
        feat_out =  self.net(torch.cat([frames_pred.pts3d, frames_pred.feats], dim=-1))

        if frames_pred.feat is not None:
            frames_pred.feat = torch.cat([feat_out, frames_pred.feat], dim=-1)
        else:
            frames_pred.feat = feat_out

        return frames_gt, frames_pred
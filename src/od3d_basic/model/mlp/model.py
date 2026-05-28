import logging

logger = logging.getLogger(__name__)
import torch.nn as nn
from od3d_basic.model.model import OD3D_Model, register_model
from typing import List
from omegaconf import DictConfig
import torch

class OD3D_Norm_Detach(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # ... x F
        x = x / ((x.norm(dim=-1, keepdim=True) + 1e-10).detach())
        return x


class OD3D_Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # ... x F
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-10)
        return x


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "norm":
        return OD3D_Norm()
    elif name == "norm_detach":
        return OD3D_Norm_Detach()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    elif name == "none":
        return nn.Identity()
    else:
        raise NotImplementedError


@register_model("MLP")
class MLP(OD3D_Model):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        dropout,
        in_dim = None,
        in_dims: List = None,
        activation = None,
        bias = False,
        out_dim = None,
        out_dims = None,
    ):
        super().__init__()

        if in_dims is not None:
            self.in_dim = in_dims[-1]
        else:
            self.in_dim = in_dim

        # def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        #     super().__init__()
        num_layers = num_layers
        hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.out_dims = out_dims
        if self.out_dim is None and self.out_dims is None:
            msg = f"Either out_dim or out_dims must not be None for MLP."
            raise ValueError(msg)
        
        if self.out_dim is None:
            self.out_dim = sum(self.out_dims)
        if self.out_dims is None:
            self.out_dims = [self.out_dim]

        dropout = dropout
        activation = activation
        self.bias = bias
        assert num_layers >= 1
        nets = []
        for out_dim in self.out_dims:
            if num_layers == 1:
                network = [nn.Linear(self.in_dim, out_dim, bias=self.bias)]
            else:
                network = [nn.Linear(self.in_dim, hidden_dim, bias=self.bias)]
                for _ in range(num_layers - 2):
                    network += [
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim, bias=self.bias),
                    ]
                    if dropout:
                        network += [nn.Dropout(dropout)]
                network += [
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim, bias=self.bias),
                ]
            if activation is not None:
                network += [get_activation(activation)]
            nets.append(nn.Sequential(*network))
        self.nets = nn.ModuleList(nets)

    def forward(self, frames_gt, frames_pred=None):
        # if len(x.featmaps) > 1:
        #     logger.warning("MLP head only process last feature map.")

        x_res = frames_pred.feat  # [-1].flatten(1)  # BxF

        x_outs = []
        for n in range(len(self.nets)):
            x_outs.append(self.nets[n](x_res))

        x_out = torch.cat(x_outs, dim=-1)

        frames_pred.feat = x_out
        return frames_gt, frames_pred

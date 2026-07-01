import logging

logger = logging.getLogger(__name__)
import torch.nn as nn

# from od3d.models.heads.head import OD3D_Head
from typing import List
from omegaconf import DictConfig
import torch
from o3b.model.model import register_model
from o3b.model.mlp import MLP

class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, scalar=1, dim=-1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `scalar` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = scalar * (2.0 ** torch.arange(n_harmonic_functions))
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        if self.dim != -1:
            x = x.transpose(self.dim, -1).contiguous()

        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)

        if self.dim != -1:
            embed = embed.transpose(self.dim, -1).contiguous()
            x = x.transpose(self.dim, -1).contiguous()

        return torch.cat((embed.sin(), embed.cos()), dim=self.dim)


@register_model("CoordMLP")
class CoordMLP(MLP):
    def __init__(
        self,
        in_dims: List = None,
        in_dim = None,
        in_upsample_scales: List = None,   # ignored; accepted for od3d API compatibility
        config: DictConfig = None,         # MLP config: num_layers, hidden_dim, out_dim, dropout, activation, symmetrize
        symmetrize = None,
        query_dim = 3,
        n_harmonic_functions = 10,
        embedder_scalar = 2.8274,
        embed_concat_pts = True,
    ):
        # Extract MLP params from config dict if provided (od3d-style API)
        if config is not None:
            num_layers  = config.get("num_layers",  5)
            hidden_dim  = config.get("hidden_dim",  256)
            out_dim     = config.get("out_dim",     None)
            dropout     = config.get("dropout",     0)
            activation  = config.get("activation",  None)
            if symmetrize is None:
                symmetrize = config.get("symmetrize", True)
        else:
            num_layers, hidden_dim, out_dim, dropout, activation = 5, 256, None, 0, None

        if symmetrize is None:
            symmetrize = True

        feat_dim = in_dims[-1] if in_dims is not None else (in_dim or 0)

        if n_harmonic_functions > 0:
            embed_dim = query_dim * 2 * n_harmonic_functions
            if embed_concat_pts:
                embed_dim += query_dim
        else:
            embed_dim = query_dim

        mlp_in_dim = feat_dim + embed_dim

        # nn.Module.__init__ must happen before any nn.Module is assigned as attribute
        super().__init__(
            in_dim=mlp_in_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
        )

        self.embed_dim = embed_dim
        self.embed_concat_pts = embed_concat_pts
        self.symmetrize = symmetrize

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, embedder_scalar)
        else:
            self.embedder = None

    def forward(self, frames_gt, frames_pred=None):

        x = frames_pred
        pts3d = x.pts3d  # BxNx3
        B, N = pts3d.shape[0], pts3d.shape[1]
        if self.symmetrize:
            # pts3d[:, :, 0] = pts3d[:, :, 0].abs() # mirror -x to +x
            pts3d_x, pts3d_y, pts3d_z = pts3d.unbind(-1)
            pts3d = torch.stack(
                [pts3d_x.abs(), pts3d_y, pts3d_z],
                -1,
            )  # mirror -x to +x

        if self.embedder is not None:
            pts3d_embed = self.embedder(pts3d)
            if self.embed_concat_pts:
                pts3d_embed = torch.cat([pts3d, pts3d_embed], -1)
        else:
            pts3d_embed = pts3d

        if x.latent is not None:
            x_latent = x.latent
            x_latent_mu = x.latent_mu
            x_latent_logvar = x.latent_logvar
        else:
            x_latent = x.feat  # [-1].flatten(1)  # BxF
            x_latent_mu = x.feat_mu
            x_latent_logvar = x.feat_logvar

        if x_latent is not None:
            feats = torch.cat(
                [
                    pts3d_embed,
                    x_latent[:, None].expand(
                        *pts3d_embed.shape[:2],
                        x_latent.shape[-1],
                    ),
                ],
                dim=-1,
            )  # BxNxE+F
        else:
            feats = pts3d_embed  # BxNxE+F
        
        frames_pred.feat = feats.reshape(-1, feats.shape[-1])
        frames_gt, frames_pred = super().forward(frames_gt=frames_gt, frames_pred=frames_pred)
        frames_pred.feat = frames_pred.feat.reshape(B, N, -1)

        if x_latent is not None:
            frames_pred.latent = x_latent  # .reshape(B, -1)
            frames_pred.latent_mu = x_latent_mu  # .reshape(B, -1)
            frames_pred.latent_logvar = x_latent_logvar  # .reshape(B, -1)

        return frames_gt, frames_pred


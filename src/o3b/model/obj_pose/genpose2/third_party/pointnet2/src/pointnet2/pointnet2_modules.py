import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils
from typing import List


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, return_idx=False) \
        -> "(torch.Tensor, torch.Tensor) | (torch.Tensor, torch.Tensor, torch.Tensor)":
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
            new_idx: (B, npoint) tensor of indices
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        idx = None
        if new_xyz is None:
            if self.npoint is not None:
                idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                new_xyz = pointnet2_utils.gather_operation(
                    xyz_flipped,
                    idx
                ).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        if return_idx:
            return new_xyz, torch.cat(new_features_list, dim=1), idx
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool', instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method, instance_norm=instance_norm
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)

        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)





def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)

MSG_CFG = {
    'NPOINTS': [512, 256, 128, 64],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]]],
    'FP_MLPS': [[64, 64], [128, 128], [256, 256], [512, 512]],
    'CLS_FC': [128],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG = {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]],
             [[512, 512], [512, 512]]],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG_Dense = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[32, 64], [16, 32], [8, 16], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}


########## Best before 29th April ###########
ClsMSG_CFG_Light = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}


ClsMSG_CFG_Lighter= {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01], [0.02], [0.04], [0.08], [None]],
    'NSAMPLE': [[64], [32], [16], [8], [None]],
    'MLPS': [[[32, 32, 64]],
             [[64, 64, 128]],
             [[128, 196, 256]],
             [[256, 256, 512]],
             [[512, 512, 1024]]],
    'DP_RATIO': 0.5,
}

SELECTED_PARAMS = ClsMSG_CFG_Light
# if cfg.pointnet2_params == 'light':
#     SELECTED_PARAMS = ClsMSG_CFG_Light
# elif cfg.pointnet2_params == 'lighter':
#     SELECTED_PARAMS = ClsMSG_CFG_Lighter
# elif cfg.pointnet2_params == 'dense':
#     SELECTED_PARAMS = ClsMSG_CFG_Dense
# else:
#     raise NotImplementedError


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(MSG_CFG['NPOINTS'].__len__()):
            mlps = MSG_CFG['MLPS'][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=MSG_CFG['NPOINTS'][k],
                    radii=MSG_CFG['RADIUS'][k],
                    nsamples=MSG_CFG['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(MSG_CFG['FP_MLPS'].__len__()):
            pre_channel = MSG_CFG['FP_MLPS'][k + 1][-1] if k + 1 < len(MSG_CFG['FP_MLPS']) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + MSG_CFG['FP_MLPS'][k])
            )

        cls_layers = []
        pre_channel = MSG_CFG['FP_MLPS'][0][-1]
        for k in range(0, MSG_CFG['CLS_FC'].__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, MSG_CFG['CLS_FC'][k], bn=True))
            pre_channel = MSG_CFG['CLS_FC'][k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class Pointnet2ClsMSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        for k in range(SELECTED_PARAMS['NPOINTS'].__len__()):
            mlps = SELECTED_PARAMS['MLPS'][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS['NPOINTS'][k],
                    radii=SELECTED_PARAMS['RADIUS'][k],
                    nsamples=SELECTED_PARAMS['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            channel_in = channel_out


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_features[-1].squeeze(-1)


class Pointnet2ClsMSGFus(nn.Module):
    """
    Modified from Pointnet2ClsMSG:
    concatenate input features with each layer, to enable full feature fusion
    """
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        for k in range(SELECTED_PARAMS['NPOINTS'].__len__()):
            mlps = SELECTED_PARAMS['MLPS'][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS['NPOINTS'][k],
                    radii=SELECTED_PARAMS['RADIUS'][k],
                    nsamples=SELECTED_PARAMS['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            channel_in = channel_out + input_channels


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        # xyz:      bs * npoints * 3
        # features: bs * F * npoints

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i != 0:
                l_features[i] = torch.concatenate([l_features[i], features], dim=1) # concatenate
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], return_idx=True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            if idx != None:
                features = torch.gather(
                    features, 2,
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(-1, features.shape[1], -1)
                ) # only keep features of remaining points
            else:
                assert i == len(self.SA_modules) - 1
        return l_features[-1].squeeze(-1)


if __name__ == "__main__":
    pass

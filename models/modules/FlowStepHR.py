import torch
from torch import nn as nn

from models.modules import ActNorms, Permutations
from models.modules.Affine import SparseAffine, SparseAffineHR, CondSparseAffineHR, FT


class FlowCell(nn.Module):
    def __init__(self, node_num, in_channels=4, actnorm_scale=1.0, LU_decomposed=False):
        super().__init__()

        # 1. actnorm
        self.actnorm = ActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute # todo: maybe hurtful for downsampling; presever the structure of downsampling
        self.permute = Permutations.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        # self.permute = None
        # self.actnorm = None
        # 3. coupling
        self.Ft = FT(node_num, channel=in_channels)
        self.affine = SparseAffineHR(node_num, channel=in_channels)
        self.condaffine = CondSparseAffineHR(node_num, channel=in_channels)

    def forward(self, z, lr_list=None, u=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, lr_list, logdet)
        else:
            return self.reverse_flow(z, lr_list)

    def normal_flow(self, z, lr_list=None, logdet=None):
        # 1. actnorm
        if self.actnorm is not None:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        if self.permute is not None:
            z, logdet = self.permute(z, logdet=logdet, reverse=False)

        # 3. coupling
        ft = self.Ft(lr_list)

        z, logdet = self.affine(z, lr_list=lr_list, logdet=logdet, reverse=False, ft=ft)

        # 3. conditional affine
        z, logdet = self.condaffine(z, logdet=logdet, reverse=False, ft=ft)
        return z, logdet

    def reverse_flow(self, z, lr_list=None, logdet=None):

        ft = self.Ft(lr_list)

        # 1. conditional affine
        z, logdet = self.condaffine(z, logdet=logdet, reverse=True, ft=ft)

        # 2.coupling
        z, _ = self.affine(z, lr_list, reverse=True, ft=ft)

        # 3. permute
        if self.permute is not None:
            z, _ = self.permute(z, reverse=True)

        # 4. actnorm
        if self.actnorm is not None:
            z, _ = self.actnorm(z, reverse=True)

        return z, logdet
import torch
from torch import nn as nn

from utils.util import opt_get
from models.modules import ActNorms, Permutations
from models.modules.Affine import SparseAffine


class FlowCell(nn.Module):
    def __init__(self, node_num, in_channels=4, actnorm_scale=1.0, LU_decomposed=False):
        super().__init__()

        # 1. actnorm
        self.actnorm = ActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute # todo: maybe hurtful for downsampling; presever the structure of downsampling
        # self.permute = Permutations.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        self.permute = None
        # self.actnorm = None
        # 3. coupling
        self.affine = SparseAffine(node_num)

    def forward(self, z, u=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, logdet)
        else:
            return self.reverse_flow(z)

    def normal_flow(self, z, logdet=None):
        # 1. actnorm
        if self.actnorm is not None:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        if self.permute is not None:
            z, logdet = self.permute(z, logdet=logdet, reverse=False)

        # 3. coupling
        z, logdet = self.affine(z, logdet=logdet, reverse=False)

        return z, logdet

    def reverse_flow(self, z, logdet=None):
        # 1.coupling
        z, _ = self.affine(z, reverse=True)

        # 2. permute
        if self.permute is not None:
            z, _ = self.permute(z, reverse=True)

        # 3. actnorm
        if self.actnorm is not None:
            z, _ = self.actnorm(z, reverse=True)

        return z, logdet
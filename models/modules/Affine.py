import torch
from torch import nn
from torch.nn import init
import copy
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter
import math
from models.modules.mnist_conv import Net
from models.modules import thops
import spconv.pytorch as spconv
from models.modules.convGRUSE import ConvGRU
from models.modules.Basic import SplitHR, Conv2d, Conv2dZeros
# torch.cuda.manual_seed_all(1)


class SparseAffine(nn.Module):
    def __init__(self, node_num):
        super().__init__()
        self.node_num = node_num
        self.affine = Net(self.node_num, self.node_num)
        self.linear = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.Softmax(dim=-1),
        )

    def forward(self, input, logdet=None, reverse=False):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num*node_fea): rrdbResults
        :param gt: FloatTensor (batch_size, 1, node_num*node_fea): z
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        window_size = 1 default
        """
        batch_size, channel, node_num = input.size()[0: 3]
        # logs = []
        # for b in range(batch_size):
        #     temp = self.affine(input[b])
        #     temp = temp.view(batch_size, channel, node_num, -1)
        #     if b == 0:
        #         logs = temp
        #     else:
        #         logs = torch.cat((logs, temp), dim=0)
        input = input.view(1, batch_size, channel, node_num, node_num)
        logs = self.affine(input)
        logs = logs.view(batch_size, channel, node_num, -1)
        input = input.view(batch_size, channel, node_num, -1)
        if not reverse:
            z = input * torch.exp(self.linear(logs)) + logs  # should have shape batchsize, n_channels, 1, 1
        else:
            z = (input - logs) * torch.exp(-self.linear(logs))
        '''loss'''
        dlogdet = self.get_logdet(self.linear(logs))
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        else:
            logdet = dlogdet
        return z, logdet

    def get_logdet(self, scale):
        return thops.sum(scale, dim=[1, 2, 3])


class FT(nn.Module):
    def __init__(self, node_num, channel):
        super().__init__()

        self.channel = channel
        self.node_num = node_num
        self.affine = Net(self.node_num, self.node_num)
        self.before = channel - 1
        self.affineHR = Net(1, self.channel, hidden_features=32)
        self.linear = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.Softmax(dim=-1),
        )
        self.linearLR = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.ReLU(),
        )
        self.convgru_features = 1
        dtype = torch.cuda.FloatTensor
        self.convGRU = ConvGRU(input_size=(self.node_num, self.node_num),
                               input_dim=1,
                               hidden_dim=[self.convgru_features, 1],
                               kernel_size=(5, 5),
                               num_layers=2,
                               dtype=dtype,
                               batch_first=True,
                               bias=True,
                               return_all_layers=False)

    def forward(self, lr_list):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num*node_fea): rrdbResults
        :param gt: FloatTensor (batch_size, 1, node_num*node_fea): z
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        window_size = 1 default
        """
        # print('in=========================')
        batch_size, channel, node_num = lr_list.size()[0: 3]
        channel = self.channel
        te_output_list = lr_list.view(batch_size, self.before, 1, node_num, -1)
        _, hn = self.convGRU(te_output_list)
        LR_gru = hn[0][0]
        LR_gru = LR_gru.view(1, -1, node_num, node_num, 1)
        LR = self.affineHR(LR_gru)
        LR = LR.view(batch_size, channel, node_num, -1)
        ft = self.linearLR(LR)
        return ft

    def get_logdet(self, scale):
        return thops.sum(scale, dim=[1, 2, 3])


class SparseAffineHR(nn.Module):
    def __init__(self, node_num, channel):
        super().__init__()
        self.node_num = node_num
        self.channel = channel
        self.affine = Net(self.node_num, self.node_num)
        self.before = channel - 1
        self.affineHR = Net(1, channel, hidden_features=32)
        self.linear = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.Softmax(dim=-1),
        )
        self.linearLR = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=int(self.node_num * self.node_num),
            hidden_size=int(32),
            num_layers=1,
            batch_first=True
        )
        self.convgru_features = 1
        dtype = torch.cuda.FloatTensor
        self.convGRU = ConvGRU(input_size=(self.node_num, self.node_num),
                               input_dim=1,
                               hidden_dim=[self.convgru_features, 1],
                               kernel_size=(5, 5),
                               num_layers=2,
                               dtype=dtype,
                               batch_first=True,
                               bias=True,
                               return_all_layers=False)

    def forward(self, input, lr_list=None, logdet=None, reverse=False, ft=None):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num*node_fea): rrdbResults
        :param gt: FloatTensor (batch_size, 1, node_num*node_fea): z
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        window_size = 1 default
        """
        # print('in=========================')
        batch_size, channel, node_num = input.size()[0: 3]
        input = input.view(1, batch_size, channel, node_num, node_num)
        logs = self.affine(input)
        logs = logs.view(batch_size, channel, node_num, -1)
        input = input.view(batch_size, channel, node_num, -1)

        if not reverse:
            z = input * torch.exp(self.linear(logs)) + ft  # should have shape batchsize, n_channels, 1, 1
        else:
            z = (input - ft) * torch.exp(-self.linear(logs))
        '''loss'''
        dlogdet = self.get_logdet(self.linear(logs))
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        else:
            logdet = dlogdet

        return z, logdet

    def get_logdet(self, scale):
        return thops.sum(scale, dim=[1, 2, 3])


class CondSparseAffineHR(nn.Module):
    def __init__(self, node_num, channel):
        super().__init__()
        self.node_num = node_num
        self.channel = channel
        self.affine = Net(self.node_num, self.node_num)
        self.before = self.channel - 1
        self.affineHR = Net(1, channel, hidden_features=32)
        self.linear = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.Softmax(dim=-1),
        )
        self.linearLR = nn.Sequential(
            nn.Linear(self.node_num, self.node_num),
            nn.ReLU(),
        )
        self.split = SplitHR(num_channels_split=self.channel // 2)
        self.relu = nn.ReLU()

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, ft):
        h = ft
        shift, scale = thops.split_feature(h)
        scale = (torch.sigmoid(scale))
        return scale, shift

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(ft)
            z2 = z2 + shift
            z2 = z2 * scale
            if logdet is not None:
                """
                logs is log_std of `mean of channels`
                so we need to multiply pixels
                """
                logdet = logdet + self.get_logdet(scale)
            else:
                logdet = self.get_logdet(scale)

            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(ft)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            if logdet is not None:
                """
                logs is log_std of `mean of channels`
                so we need to multiply pixels
                """
                logdet = logdet - self.get_logdet(scale)
            else:
                logdet = - self.get_logdet(scale)
            output = z
        return output, logdet

    def get_logdet(self, scale):
        return thops.sum(scale, dim=[1, 2, 3])



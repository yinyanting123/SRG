import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from models.modules import thops


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std).cuda()
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std).cuda()
        return eps


class UnSplit(nn.Module):
    def __init__(self, channel, num_channels_split, level):
        super().__init__()
        self.channel = channel
        self.num_channels_split = num_channels_split
        self.level = level
        self.relu = nn.ReLU()

    def forward(self, z, reverse=False):
        if reverse:
            z1 = z.sum(axis=1) / z.size(1)
            return z[:, :self.num_channels_split, ...], z[:, self.num_channels_split:, ...]
        else:
            node_num = z.size()[-1]
            channel = self.channel
            batch_size = z.size()[0]
            eps = GaussianDiag.sample_eps((batch_size, channel - 1, node_num, node_num), 1, seed=1)
            result = torch.cat((z, eps), dim=1)
            return result


class UnSplitHR(nn.Module):
    def __init__(self, num_channels_split, level):
        super().__init__()
        self.num_channels_split = num_channels_split
        self.level = level
        self.relu = nn.ReLU()

    def forward(self, z, reverse=False):
        if reverse:
            z1 = z.sum(axis=1) / z.size(1)
            z = self.relu(z)
            return z[:, :self.num_channels_split, ...], z[:, self.num_channels_split:, ...]
        else:
            node_num = z.size()[-1]
            channel = z.size()[1] * 4
            batch_size = z.size()[0]
            eps = GaussianDiag.sample_eps((batch_size, channel - 1, node_num, node_num), 1)
            result = torch.cat((z, eps), dim=1)
            return result


class SplitHR(nn.Module):
    def __init__(self, num_channels_split, level=0):
        super().__init__()
        self.num_channels_split = num_channels_split
        self.level = level
        self.relu = nn.ReLU()

    def forward(self, z, reverse=False):
        if not reverse:
            return z[:, :self.num_channels_split, ...], z[:, self.num_channels_split:, ...]
        else:
            node_num = z.size()[-1]
            channel = z.size()[1] * 4
            batch_size = z.size()[0]
            eps = GaussianDiag.sample_eps((batch_size, channel - 1, node_num, node_num), 1)
            result = torch.cat((z, eps), dim=1)
            return result


def squeeze2d(input, factor=2, factor2=1):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    x = input.view(B, C, H // factor, factor, W // factor2, factor2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor2, H // factor, W // factor2)
    return x


def unsqueeze2d(input, factor=2, factor2=1):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1 and factor2 == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    x = input.view(B, C // (factor * factor2), factor, factor2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor * factor2), H * factor, W * factor2)
    return x


def squeeze2d_hr(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    H_new = H // factor
    W_new = W // factor
    C_new = C * factor * factor
    x = torch.zeros((B, C_new, H_new, W_new)).cuda()

    temp = np.arange(H_new) * 2
    temp = temp.reshape(-1, 1)
    index_list = temp * np.ones((1, H_new))

    i = index_list.reshape((1, -1))
    j = index_list.T.reshape((1, -1))
    i_1 = i + 1
    j_1 = j + 1

    index_1 = np.concatenate([i, j], axis=0)
    index_1 = tuple(index_1)
    index_2 = np.concatenate([i_1, j], axis=0)
    index_2 = tuple(index_2)
    index_3 = np.concatenate([i, j_1], axis=0)
    index_3 = tuple(index_3)
    index_4 = np.concatenate([i_1, j_1], axis=0)
    index_4 = tuple(index_4)

    for t in range(B):
        temp = input[t][0][index_1]
        temp = temp.reshape(H_new, -1)
        x[t][0] = temp

        temp = input[t][0][index_2]
        temp = temp.reshape(H_new, -1)
        x[t][1] = temp

        temp = input[t][0][index_3]
        temp = temp.reshape(H_new, -1)
        x[t][2] = temp

        temp = input[t][0][index_4]
        temp = temp.reshape(H_new, -1)
        x[t][3] = temp
    return x


def squeeze2d_multi_channel(input, factor=2, factor2=2):
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor2 == 0
    H_new = H // factor
    W_new = W // factor2
    C_new = C * factor * factor2
    x = torch.zeros((B, C_new, H_new, W_new)).cuda()

    temp = np.arange(H_new) * 2
    temp = temp.reshape(-1, 1)
    index_list = temp * np.ones((1, H_new))

    i = index_list.reshape((1, -1))
    j = index_list.T.reshape((1, -1))
    i_1 = i + 1
    j_1 = j + 1

    index_1 = np.concatenate([i, j], axis=0)
    index_1 = tuple(index_1)
    index_2 = np.concatenate([i_1, j], axis=0)
    index_2 = tuple(index_2)
    index_3 = np.concatenate([i, j_1], axis=0)
    index_3 = tuple(index_3)
    index_4 = np.concatenate([i_1, j_1], axis=0)
    index_4 = tuple(index_4)

    for t in range(B):
        temp = input[t][0][index_1]
        temp = temp.reshape(H_new, -1)
        x[t][0] = temp

        temp = input[t][0][index_2]
        temp = temp.reshape(H_new, -1)
        x[t][1] = temp

        temp = input[t][0][index_3]
        temp = temp.reshape(H_new, -1)
        x[t][2] = temp

        temp = input[t][0][index_4]
        temp = temp.reshape(H_new, -1)
        x[t][3] = temp
    return x


def unsqueeze2d_hr(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    H_new = H * factor
    W_new = W * factor
    C_new = C // (factor2)
    x = torch.zeros((B, C_new, H_new, W_new)).cuda()
    for t in range(B):
        for i in range(H):
            for j in range(W):
                for c in range(4):
                    if c == 0:
                        x[t][0][2 * i][2 * j] = input[t][c][i][j]
                    elif c == 1:
                        x[t][0][2 * i + 1][2 * j] = input[t][c][i][j]
                    elif c == 2:
                        x[t][0][2 * i][2 * j + 1] = input[t][c][i][j]
                    elif c == 3:
                        x[t][0][2 * i + 1][2 * j + 1] = input[t][c][i][j]
    # x = input.view(B, C // factor2, factor, factor, H, W)
    # x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    # x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


def unsqueeze2dCons(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    H_new = H * factor
    W_new = W * factor
    C_new = C // (factor2)
    x = torch.zeros((B, C_new, H_new, W_new)).cuda()
    cons = torch.zeros((B, C_new, H_new, W_new)).cuda()
    A = input
    s3 = None
    s2 = None
    s1 = None
    s0 = None
    for b in range(B):
        if b == 0:
            s3 = torch.ones_like(A[0][0])
            s2 = (A[0][0] + 1) / 2
            s1 = (A[0][0] + A[0][1] + 1) / 3
            s0 = (A[0][0] + A[0][1] + A[0][2] + 1) / 4

            s3 = s3.view(1, H, W)
            s2 = s2.view(1, H, W)
            s1 = s1.view(1, H, W)
            s0 = s0.view(1, H, W)
        else:
            s3_temp = torch.ones_like(A[0][0])
            s2_temp = (A[0][0] + 1) / 2
            s1_temp = (A[0][0] + A[0][1] + 1) / 3
            s0_temp = (A[0][0] + A[0][1] + A[0][2] + 1) / 4

            s3_temp = s3_temp.view(1, H, W)
            s2_temp = s2_temp.view(1, H, W)
            s1_temp = s1_temp.view(1, H, W)
            s0_temp = s0_temp.view(1, H, W)

            s3 = torch.cat((s3, s3_temp), dim=0)
            s2 = torch.cat((s2, s2_temp), dim=0)
            s1 = torch.cat((s1, s1_temp), dim=0)
            s0 = torch.cat((s0, s0_temp), dim=0)

    temp = np.arange(H) * 2
    temp = temp.reshape(-1, 1)
    index_list = temp * np.ones((1, H))

    i = index_list.reshape((1, -1))
    j = index_list.T.reshape((1, -1))
    i_1 = i + 1
    j_1 = j + 1

    index_1 = np.concatenate([i, j], axis=0)
    index_1 = tuple(index_1)
    index_2 = np.concatenate([i_1, j], axis=0)
    index_2 = tuple(index_2)
    index_3 = np.concatenate([i, j_1], axis=0)
    index_3 = tuple(index_3)
    index_4 = np.concatenate([i_1, j_1], axis=0)
    index_4 = tuple(index_4)
    for t in range(B):
        a1 = input[t][0]
        a1 = a1.view(H*W)
        s3 = s3.view(H*W)
        x[t][0][index_1] = a1
        cons[t][0][index_1] = s3

        a2 = input[t][1]
        a2 = a2.view(H*W)
        s2 = s2.view(H*W)
        x[t][0][index_2] = a2
        cons[t][0][index_2] = s2

        a3 = input[t][2]
        a3 = a3.view(H*W)
        s1 = s1.view(H*W)
        x[t][0][index_3] = a3
        cons[t][0][index_3] = s1

        a4 = input[t][3]
        a4 = a4.view(H*W)
        s0 = s0.view(H*W)
        x[t][0][index_4] = a4
        cons[t][0][index_4] = s0

    # for t in range(B):
    #     for i in range(H):
    #         for j in range(W):
    #             for c in range(4):
    #                 if c == 0:
    #                     x[t][0][2 * i][2 * j] = input[t][c][i][j]
    #                     cons[t][0][2 * i][2 * j] = s3[t][i][j]
    #                 elif c == 1:
    #                     x[t][0][2 * i + 1][2 * j] = input[t][c][i][j]
    #                     cons[t][0][2 * i + 1][2 * j] = s2[t][i][j]
    #                 elif c == 2:
    #                     x[t][0][2 * i][2 * j + 1] = input[t][c][i][j]
    #                     cons[t][0][2 * i][2 * j + 1] = s1[t][i][j]
    #                 elif c == 3:
    #                     x[t][0][2 * i + 1][2 * j + 1] = input[t][c][i][j]
    #                     cons[t][0][2 * i + 1][2 * j + 1] = s0[t][i][j]
    # x = input.view(B, C // factor2, factor, factor, H, W)
    # x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    # x = x.view(B, C // (factor2), H * factor, W * factor)

    return x, cons


class UnSqueezeLayer(nn.Module):
    def __init__(self, factor, factor2):
        super().__init__()
        self.factor = factor
        self.factor2 = factor2

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = unsqueeze2d(input, self.factor, self.factor2)
            return output, logdet
        else:
            output = squeeze2d(input, self.factor, self.factor2)
            return output, logdet


class UnSqueezeConsLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output, cons = unsqueeze2dCons(input, self.factor)
            return output, cons, logdet
        else:
            output = squeeze2d_hr(input, self.factor)
            return output, logdet


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)
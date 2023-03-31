import torch
from torch import nn as nn
import torch.nn.functional as F
from models.modules import Basic, FlowStepHR


class FlowNet(nn.Module):
    def __init__(self, image_shape, channel, opt=None):
        assert image_shape[0] == 1 or image_shape[0] == 3
        super().__init__()
        self.C, H, W = image_shape
        self.channel = channel

        # construct flow
        self.layers = nn.ModuleList()
        self.output_shapes = []

        # 1. UnSplite
        self.layers.append(Basic.UnSplit(channel=self.channel, num_channels_split=1, level=0))
        self.output_shapes.append([-1, self.C, H, W])

        # 2. flow cell
        self.layers.append(FlowStepHR.FlowCell(node_num=H, in_channels=channel, actnorm_scale=1.0, LU_decomposed=False))
        self.output_shapes.append([-1, self.C, H, W])

        # 3. UnSqueeze
        self.layers.append(Basic.UnSqueezeLayer(factor=2, factor2=self.channel // 2))  # may need a better way for squeezing
        self.C, H, W = self.C * self.channel, H, W // 2
        self.output_shapes.append([-1, self.C, H, W])

        self.H = H
        self.W = W
        self.scaleH = image_shape[0] / H
        self.scaleW = image_shape[1] / W

    def forward(self, hr=None, lr_list=None, z=None, u=None, eps_std=None, logdet=None, reverse=False, training=True):
        if not reverse:
            return self.normal_flow(hr, lr_list=lr_list, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z, lr_list=lr_list, eps_std=eps_std, training=training)

    '''
    lr -> hr
    input: batch * channel * node_num * node_num
    1*n*n -> 4*n*n (UnSplit) -> 4*n*n (Flow) -> (n*2)*(n*2) (UnSqueeze)
    '''
    def normal_flow(self, z, lr_list=None, logdet=None, training=True):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, FlowStepHR.FlowCell):
                z, logdet = layer(z, lr_list=lr_list, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.UnSqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.UnSplit):
                z = layer(z, reverse=False)
        return z, logdet

    '''
    hr -> lr
    (n*2)*(n*2) -> 4*n*n (Squeeze) -> 4*n*n (Flow) -> 1*n*n (Split)
    '''
    def reverse_flow(self, z, lr_list=None, u=None, eps_std=None, training=True):
        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            if isinstance(layer, FlowStepHR.FlowCell):
                z, _ = layer(z, lr_list=lr_list, reverse=True)
            elif isinstance(layer, Basic.UnSqueezeLayer):
                z, _ = layer(z, reverse=True)
            elif isinstance(layer, Basic.UnSplit):
                    z, a1 = layer(z, reverse=True)
        return z





















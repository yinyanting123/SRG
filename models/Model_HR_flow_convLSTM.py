'''
degree * weight ==> norm ==> nolear ==> added in convGRU
'''
import torch
from torch import nn
from models.convGRU_One import ConvGRU
from models.convLSTM import ConvLSTM
import os
import numpy as np
from models.modules.mnist_conv_1channel import Net
from models.modules.flow_HR import FlowNet
import torch.nn.functional as F
torch.cuda.manual_seed_all(1)
from torch.distributions import MultivariateNormal as MVN


class WA_BaseModel(nn.Module):
    def __init__(self,
                 window_size,
                 node_num,
                 out_features,
                 convgru_features,
                 convGRU_kernel,
                 channel,
                 opt):
        super(WA_BaseModel, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.node_num = node_num
        self.out_features = out_features
        self.convgru_features = convgru_features
        '''flow'''
        self.flow = FlowNet((1, self.node_num, self.node_num), self.channel)
        '''link para'''
        self.sig = nn.Sigmoid()
        '''spconv setting'''
        self.spconv = Net(self.node_num * channel // 2, out_features)
        '''convGRU'''
        dtype = torch.cuda.FloatTensor
        self.convGRU = ConvGRU(input_size=(node_num*2, out_features),
                               input_dim=1,
                               hidden_dim=[self.convgru_features, 1],
                               kernel_size=(int(convGRU_kernel), int(convGRU_kernel)),
                               num_layers=2,
                               dtype=dtype,
                               batch_first=True,
                               bias=True,
                               return_all_layers=False)
        self.convLSTM = ConvLSTM(input_dim=1,
                                 hidden_dim=[self.convgru_features, 1],
                                 kernel_size=(int(convGRU_kernel), int(convGRU_kernel)),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.out_features, node_num*channel // 2),
            nn.Sigmoid()
        )
        self.mse = nn.MSELoss()
        self.criterion = nn.BCELoss()
        self.ffnLR = (nn.Sigmoid())
        '''loss '''
        init_noise_sigma = 8.0
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma)).cuda()

    def convert_sp_mat_to_sp_tensor(self, X):
        # scipy稠密矩阵转换为scipy的稀疏矩阵方法为scipy.sparse，scipy的稀疏矩阵转为稠密矩阵的方法，直接.todense()
        # pytorch的稀疏矩阵转为稠密矩阵.to_dense()，稠密矩阵转稀疏矩阵torch.sparse.FloatTensor(i, v, coo.shape)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def bmc_loss_md(self, pred, target, noise_var):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, d].
          target: A float tensor of size [batch, d].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        I = torch.eye(pred.shape[-1]).cuda()
        logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    def forward(self, HR, LR, use_HR=False, use_MSE=0, reverse=False):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num, node_num)
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        """
        channel = 1
        batch_size, window_size, node_num = HR.size()[0: 3]
        '''LR ==> HR through Flow'''
        HR_ge = []
        if not reverse:
            '''conv'''
            logdet_sum = 0
            for b in range(batch_size):
                HR_ge_temp = []
                for w in range(window_size):
                    LR_input = LR[b][w].reshape(1, 1, self.node_num, self.node_num)
                    gap = max(self.channel - w - 1, 0)
                    LR_before = []
                    for g in range(gap):
                        temp = LR[b][0].reshape(1, 1, self.node_num, self.node_num)
                        if g == 0:
                            LR_before = temp
                        else:
                            LR_before = torch.cat((temp, LR_before), dim=1)
                    for before_i in range(w - (self.channel - 1) + gap, w):
                        temp = LR[b][before_i].reshape(1, 1, self.node_num, self.node_num)
                        if len(LR_before) == 0:
                            LR_before = temp
                        else:
                            LR_before = torch.cat((LR_before, temp), dim=1)
                    HR_temp, logdet = self.flow(hr=LR_input, lr_list=LR_before)
                    logdet_sum += logdet
                    HR_temp = HR_temp.view(1, 1, self.node_num*2, -1)
                    if w == 0:
                        HR_ge_temp = HR_temp
                    else:
                        HR_ge_temp = torch.cat((HR_ge_temp, HR_temp), dim=1)
                if b == 0:
                    HR_ge = HR_ge_temp
                else:
                    HR_ge = torch.cat((HR_ge, HR_ge_temp), dim=0)
            HR_ge = self.sig(HR_ge)
            if use_MSE == 0:
                loss_generate = self.mse(HR, HR_ge)
            elif use_MSE == 1:
                loss_generate = self.criterion(HR, HR_ge) / (batch_size * window_size)
                # loss_generate += 0.001 * logdet_sum[0]
            else:
                # noise_var = self.noise_sigma ** 2
                # pred = HR_ge
                # target = HR
                # pred = pred.view(batch_size*window_size, -1)
                # target = target.view(batch_size*window_size, -1)
                # loss_generate = self.bmc_loss_md(pred, target, noise_var)
                loss_generate = self.mse(HR, HR_ge) + self.criterion(HR, HR_ge) / (batch_size * window_size)
        else:
            HR_ge = HR
            loss_generate = torch.tensor(0).cuda()
            loss_generate = loss_generate.float()
        if use_HR:
            '''method 1'''
            te_output_list = self.spconv(HR)
        else:
            '''method 2'''
            te_output_list = self.spconv(HR_ge)
        te_output_list = te_output_list.view(batch_size, window_size, channel, node_num, -1)
        '''convGRU'''
        layer_output_list, last_state_list = self.convLSTM(te_output_list)
        '''concat'''
        convGRU_output = last_state_list[0][0]
        output = self.ffn(convGRU_output)
        output = output.view(batch_size, channel, node_num, -1)
        output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
        output = output.float()
        outputHR = output
        lr_list = []
        for b in range(batch_size):
            lr_list_remp = []
            for t in range(self.window_size-(self.channel - 1), self.window_size):
                temp = LR[b][t].reshape(1, 1, self.node_num, self.node_num)
                if len(lr_list_remp) == 0:
                    lr_list_remp = temp
                else:
                    lr_list_remp = torch.cat((lr_list_remp, temp), dim=1)
            if len(lr_list) == 0:
                lr_list = lr_list_remp
            else:
                lr_list = torch.cat((lr_list, lr_list_remp), dim=0)

        outputLR = self.flow(z=outputHR, lr_list=lr_list, reverse=True, training=False)
        outputLR = outputLR.view(batch_size, self.node_num, -1)
        outputLR = self.ffnLR(outputLR)
        # outputLR = outputHR.sum(axis=1) / outputHR.size(1)
        return output, outputLR, loss_generate


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    window_size = 10
    node_num = 2000
    batch_size = 1
    '''
    window_size
    node_num
    in_features: gcn in, used in the weight[0]
    out_features: gcn out, (node, out_features) as the input of convgru
    convgru_features: convGRU hidden[0], does appear in the result
    '''
    model = WA_BaseModel(window_size=window_size, node_num=node_num, in_features=128,
                         out_features=64, convgru_features=128)
    model = model.cuda()
    input_tensor = torch.rand(batch_size, window_size, node_num, node_num).cuda()  # (b,t,h,w)
    output = model(input_tensor)
    output = np.sum(output, axis=1)
    print(output)

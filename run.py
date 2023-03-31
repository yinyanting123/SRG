# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import torch
import contextlib
import torch.cuda.amp
import yaml
import os
import logging
from nni.utils import merge_parameter
from torch import nn
from torch.utils.data.dataloader import DataLoader
from models.Model_HR_flow_GRU import SRGModel
from utils.util import LPDataset
import torch.nn.parallel
import numpy as np
from sklearn.metrics import roc_auc_score
import nni

logger = logging.getLogger('sparsetest')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = yaml.load(open('./generate_data/baseconfig.yml'), Loader=yaml.FullLoader)
print(torch.cuda.device_count())


@contextlib.contextmanager
def identity_ctx():
    yield


def pretrain(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.BCELoss()
    mse = nn.MSELoss()
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    loss_all = torch.tensor(0).cuda()
    loss_all = loss_all.float()
    index = 0
    auc = 0
    for batch_idx, (data, target, dataLR, targetLR) in enumerate(train_loader):
        data = data.reshape(-1, window_size, node_num*2, node_num*2)
        target = target.reshape(-1, 1, node_num*2, node_num*2)
        data, target, dataLR, targetLR = data.float(), target.float(), dataLR.float(), targetLR.float()
        data, target, dataLR, targetLR = data.to(device), target.to(device), dataLR.to(device), targetLR.to(device)
        optimizer.zero_grad()
        # random
        dataLR = dataLR + torch.rand_like(dataLR) / 100000
        # train
        output, outputLR, loss_generate = model(data, dataLR)
        BCE_LR = criterion(outputLR, targetLR)
        MSE_HR = mse(output, target)
        MSE_HR = mse(output, target)
        # choose 1
        # loss = MSE_HR
        # choose 2
        loss = BCE_LR + 0.05 * MSE_HR
        if args['fp16']:
            assert loss.dtype is torch.float32
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_all += loss
        index += 1
        targetLR = targetLR.cpu().detach().numpy()
        outputLR = outputLR.cpu().detach().numpy()
        auc += roc_auc_score(np.reshape(targetLR, (-1,)), np.reshape(outputLR, (-1,)))
    loss_avg = loss_all / index
    auc_avg = auc / index
    print('Train Epoch: {} \tLoss: {:.6f} \tAUC: {:.6f}'.format(
        epoch, loss_avg.item(), auc_avg))


def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.BCELoss()
    mse = nn.MSELoss()
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    loss_all = torch.tensor(0).cuda()
    loss_all = loss_all.float()
    index = 0
    auc = 0
    for batch_idx, (data, target, dataLR, targetLR) in enumerate(train_loader):
        data = data.reshape(-1, window_size, node_num*2, node_num*2)
        target = target.reshape(-1, 1, node_num*2, node_num*2)
        data, target, dataLR, targetLR = data.float(), target.float(), dataLR.float(), targetLR.float()
        data, target, dataLR, targetLR = data.to(device), target.to(device), dataLR.to(device), targetLR.to(device)
        optimizer.zero_grad()
        # random
        dataLR = dataLR + torch.rand_like(dataLR) / 100000
        # train
        output, outputLR, loss_generate = model(data, dataLR)
        BCE_LR = criterion(outputLR, targetLR)
        MSE_HR = mse(output, target)
        MSE_HR = mse(output, target)
        loss = loss_generate
        if args['fp16']:
            assert loss.dtype is torch.float32
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_all += loss
        index += 1
        targetLR = targetLR.cpu().detach().numpy()
        outputLR = outputLR.cpu().detach().numpy()
        auc += roc_auc_score(np.reshape(targetLR, (-1,)), np.reshape(outputLR, (-1,)))
    loss_avg = loss_all / index
    auc_avg = auc / index
    print('Train Epoch: {} \tLoss: {:.6f} \tAUC: {:.6f}'.format(
        epoch, loss_avg.item(), auc_avg))


def evaluate(args, model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, dataLR, targetLR in test_loader:
            data = data.reshape(-1, window_size, node_num*2, node_num*2)
            target = target.reshape(-1, 1, node_num*2, node_num*2)
            data, target, dataLR, targetLR = data.float(), target.float(), dataLR.float(), targetLR.float()
            data, target, dataLR, targetLR = data.to(device), target.to(device), dataLR.to(device), targetLR.to(device)
            dataLR = dataLR + torch.rand_like(dataLR) / 100000
            output, outputLR, loss_generate = model(data,
                                                    dataLR,
                                                    reverse=True)
            targetLR = targetLR.cpu().detach().numpy()
            outputLR = outputLR.cpu().detach().numpy()
            auc = batch_size * roc_auc_score(np.reshape(targetLR, (-1,)), np.reshape(outputLR, (-1,)))
            correct += auc

    correct /= len(test_loader.dataset)
    nni.report_final_result(correct)
    print(
        '\nTest set: AverageAccuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--pre_epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--pre_lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=True,
                        help='For mixed precision training')
    args = parser.parse_args()
    return args


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    '''data loading'''
    train_save_path = os.path.join(base_path, 'GMA_T4_train.npy')
    val_save_path = os.path.join(base_path, 'GMA_T4_test.npy')
    train_save_path_LR = os.path.join(base_path, 'train.npy')
    val_save_path_LR = os.path.join(base_path, 'test.npy')

    train_data = LPDataset(train_save_path, train_save_path_LR, window_size)
    val_data = LPDataset(val_save_path, val_save_path_LR, window_size)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
    out_features = 64
    convgru_features = 32
    torch.cuda.empty_cache()

    coLSTM = SRGModel(window_size=window_size,
                      node_num=node_num,
                      out_features=int(out_features),
                      convgru_features=convgru_features,
                      convGRU_kernel=5,
                      channel=4)
    model = coLSTM.to(device)
    pre_optimizer = torch.optim.Adamax(model.parameters(), lr=args['pre_lr'], betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=0)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0)
    for epoch in range(1, args['pre_epochs'] + 1):
        pretrain(args, model, device, train_loader, pre_optimizer, epoch)
    # for epoch in range(1, args['epochs'] + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    evaluate(args, model, device, test_loader)


if __name__ == '__main__':
    try:
        node_num = config['node_num']
        window_size = config['window_size']
        channel = 1
        batch_size = config['batch_size']
        data_name = config['dataset']

        # path for all datasets
        path = './data/'
        base_path = path + data_name + '/'
        print(base_path)

        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

import numpy as np
import os
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import copy
from torch.utils.data import Dataset
import torch

THETA = 0.01


def load_data(filePath):
    if not os.path.exists(filePath):
        raise FileNotFoundError
    else:
        return np.load(filePath)


class LPDataset(Dataset):

    def __init__(self, path, pathLR, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.dataLR = torch.from_numpy(np.load(pathLR))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size],\
               self.dataLR[item: item + self.window_size], self.dataLR[item + self.window_size]


def get_snapshot(path, node_num):
    file = open(path, 'r', encoding='utf-8')
    snapshot = np.zeros(shape=(node_num, node_num), dtype=np.float32)
    for line in file.readlines():
        line = line.strip().split(' ')
        node1 = int(float(line[0]))
        node2 = int(float(line[1]))
        snapshot[node1, node2] = int(1)
    return snapshot

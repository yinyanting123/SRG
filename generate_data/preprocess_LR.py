import yaml
import os
import numpy as np
from utils.util import get_snapshot

# load config
config = yaml.load(open('./baseconfig.yml'), Loader=yaml.FullLoader)
data_name = config['dataset']
# build path

base_path = os.path.join('./data/' + data_name + '/')
data = np.load(base_path + data_name + '.npy')
data[data > 0] = 1

save_path = './data/' + data_name + '/'
train_save_path = os.path.join(save_path, 'train.npy')
test_save_path = os.path.join(save_path, 'test.npy')

num = len(data)
sum_num = num - config['window_size'] - 3
train_num = int(num * config['train_rate'])
test_num = int(num * config['test_rate'])

train_data = data[0: train_num - config['window_size'] - 1]
test_data = data[train_num: num]

# save data
np.save(train_save_path, train_data)
np.save(test_save_path, test_data)

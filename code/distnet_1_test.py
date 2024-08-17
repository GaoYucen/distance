#%%
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import pickle

import sys
sys.path.append('code')
from config import get_config
params, _ = get_config()

#%% Load data
file_name ="data/chengdu_directed_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLength = np.max(sdm)
sdm = sdm/maxLength
print('node number: ', sdm.shape[1])

#%% use pickle to load node2vec_embed
with open('data/node2vec_embed.pkl', 'rb') as f:
    embed = pickle.load(f)
    f.close()

# 归一化
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())

#%%
node_long_lat = pd.read_csv('data/chengdu_node-mod.txt', header=0, sep=',')
node_long_lat = np.array(node_long_lat)[:,1:3]
node_long_lat[:,0] = (node_long_lat[:,0] - node_long_lat[:,0].min()) / (node_long_lat[:,0].max() - node_long_lat[:,0].min())
node_long_lat[:,1] = (node_long_lat[:,1] - node_long_lat[:,1].min()) / (node_long_lat[:,1].max() - node_long_lat[:,1].min())
long_lat_embed_dim = params.long_lat_embed_dim

#%%
# haversine_embed_dim = 1 # 暂未考虑球面距离
embed_dim = params.embed_dim
input_dim = embed_dim + long_lat_embed_dim

#%% consturct the train, valid, test dataset
indices = []
for i in range(sdm.shape[0]):
    for j in range(sdm.shape[1]):
        if i != j:
            indices.append((i, j))

# shuffle the indices
np.random.shuffle(indices)

# split the indices
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

#%% define the mlp and parameters
hidden_dim1 = 400
hidden_dim2 = 100
hidden_dim3 = 20
output_dim = 1

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc11 = nn.Linear(input_dim, hidden_dim1)
        self.fc12 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1*2, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x1, x2):
        layer_11 = F.relu(self.fc11(x1))
        layer_12 = F.relu(self.fc12(x2))
        layer_1 = torch.cat((layer_11, layer_12), -1)
        layer_2 = F.relu(self.fc2(layer_1))
        layer_3 = F.relu(self.fc3(layer_2))
        out_layer = torch.sigmoid(self.fc4(layer_3))
        return out_layer

#%%
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print('Using mps!' if torch.backends.mps.is_built() else 'Using cpu!')

# Initialize your model
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

# Define training parameters
learning_rate = 0.001
num_epochs = 100
batch_size = 256

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
min_loss = 100

#%%
# 在测试集上测试
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)
model.load_state_dict(torch.load('param/distnet_best_chengdu.ckpt'))
model.eval()
with torch.no_grad():
    loss_list = []
    pred_list = []
    dist_list = []
    for i in tqdm(range(int(len(test_indices) / batch_size))):
        # 构造batch数据
        start = i * batch_size
        end = (i + 1) * batch_size
        x1_batch, x2_bacth, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                             np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j]) for
                                            i, j
                                            in test_indices[start:end]])

        # 存储真值
        dist_list.append(y_batch)
        # 先指定x1_batch为float32类型，再转换为tensor
        batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
        batch_x2 = torch.tensor(np.array(x2_bacth).astype(np.float32)).to(device)
        batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)

        # 前向传播
        outputs = model(batch_x1, batch_x2)
        loss = criterion(outputs, batch_y.unsqueeze(-1))

        loss_list.append(loss.item())

        outputs_cpu = outputs.cpu().numpy()
        # 存储预测值
        pred_list.append(outputs_cpu)


    test_loss = np.mean(loss_list)
    print(f'Test Loss: {test_loss:.4f}')

# 汇报预测值和真值的差异
dist_list = np.array(dist_list).flatten()
pred_list = np.array(pred_list).flatten()
# diff存储相对误差的绝对值
diff = np.abs(dist_list - pred_list)/dist_list
print(f'Mean diff: {np.mean(diff):.4f}')
print(f'Median diff: {np.median(diff):.4f}')
print(f'Max diff: {np.max(diff):.4f}')
print(f'Min diff: {np.min(diff):.4f}')
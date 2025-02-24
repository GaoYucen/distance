#%%
import pandas as pd
import numpy as np
# import dask.dataframe as dd
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
file_name ="data\chengdu_directed_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLength = np.max(sdm)
sdm = sdm/maxLength
print('node number: ', sdm.shape[1])

#%% use pickle to load node2vec_embed
with open("data/node2vec_embed.pkl", 'rb') as f:
    embed = pickle.load(f)
    f.close()

# 归一化
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())

#%%
node_long_lat = pd.read_csv("data\chengdu_node-mod.txt", header=0, sep=',')
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
        if sdm[i][j] != 0.0:
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
output_dim = 50
s = 23
r = 2
class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1*2, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output)

    def forward(self, x1, x2):
        layer_11 = F.relu(self.fc1(x1))
        layer_12 = F.relu(self.fc1(x2))
        layer_1 = torch.cat((layer_11, layer_12), 1)
        layer_2 = F.relu(self.fc2(layer_1))
        layer_3 = F.relu(self.fc3(layer_2))
        out_layer = torch.sigmoid(self.fc4(layer_3))
        out_layer_2 = torch.mean(torch.abs(out_layer[:, 25:] - out_layer[:, 0:25]), -1, keepdims=True)
        # out_layer_2 = (torch.mean(torch.abs(out_layer[:, 5:5+s] - out_layer[:, 0:s]), -1, keepdims=True) * s + torch.mean((out_layer[:, 5+s:] - out_layer[:, s:5]), -1, keepdims=True) * r)/5
        return out_layer_2



#%%
device = torch.device("cpu")


# Initialize your model
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

# Define training parameters
learning_rate = 0.001
num_epochs = 40
batch_size = 256

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
min_loss = 100
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    np.random.shuffle(train_indices)
    loss_list = []
    for i in tqdm(range(int(len(train_indices) / batch_size))):
        # 构造batch数据
        start = i * batch_size
        end = (i + 1) * batch_size
        x1_batch, x2_batch, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                             np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j]) for i, j
                                            in train_indices[start:end]])
        # 转化成tensor
        batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
        batch_x2 = torch.tensor(np.array(x2_batch).astype(np.float32)).to(device)
        batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)


        # 前向传播
        outputs = model(batch_x1, batch_x2)
        loss = criterion(outputs, batch_y.unsqueeze(-1))

        loss_list.append(loss.item())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = np.mean(loss_list)

    # 在每个epoch结束时验证模型
    model.eval()
    with torch.no_grad():
        loss_list = []
        for i in tqdm(range(int(len(valid_indices) / batch_size))):
            # 构造batch数据
            start = i * batch_size
            end = (i + 1) * batch_size
            x1_batch, x2_bacth, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                                 np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j]) for
                                                i, j
                                                in valid_indices[start:end]])
            # 转化成tensor
            batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
            batch_x2 = torch.tensor(np.array(x2_bacth).astype(np.float32)).to(device)
            batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)


            # 前向传播
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))

            loss_list.append(loss.item())

        valid_loss = np.mean(loss_list)

        # 打印train_loss和eva_loss
        print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        # 如果当前模型的准确率高于之前的最佳准确率，则保存模型
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), "param\distnet_best_chengdu_L1.ckpt")
            print('Model saved.')
            early_stop = 0
            # 如果连续多轮没有下降，提前结束
        else:
            early_stop += 1
            if early_stop > 8:
                break

print("Optimization Finished!")
end_time = time.time()
print("Training time: ", end_time - start_time)

#%%
# 在测试集上测试
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)
model.load_state_dict(torch.load("param\distnet_best_chengdu_L1.ckpt"))
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
        # 存储预测值
        pred_list.append(outputs.cpu().numpy())

    test_loss = np.mean(loss_list)
    print(f'Test Loss: {test_loss:.4f}')

# 汇报预测值和真值的差异
dist_list = np.array(dist_list).flatten()
pred_list = np.array(pred_list).flatten()
# diff存储相对误差的绝对值
abdiff=np.abs(dist_list - pred_list)
diff = np.abs(dist_list - pred_list)/dist_list
print(f'absolute Mean diff: {np.mean(abdiff):.4f}')
print(f'Mean diff: {np.mean(diff):.4f}')
print(f'Median diff: {np.median(diff):.4f}')
print(f'Max diff: {np.max(diff):.4f}')
print(f'Min diff: {np.min(diff):.4f}')
#%% 绘制re数据的分布情况
import matplotlib.pyplot as plt
import numpy as np

arange = (0, 0.2)

# 创建直方图
plt.hist(diff, bins=100, alpha=0.5, color='g', range = arange)
plt.savefig('figure/distribution_chengdu_L1.pdf')

#%% 统计>0.2的数量及百分比
count = 0
for i in diff:
    if i > 0.2:
        count += 1
print(count)
print(count/len(diff))

#%% 统计>1的数量及百分比
count = 0
for i in diff:
    if i > 1:
        count += 1
print(count)
print(count/len(diff))

#%% 绘制re数据的分布情况
import matplotlib.pyplot as plt
import numpy as np

max_re = np.max(diff)
arange = (0, max_re)

# 创建直方图
plt.hist(diff, bins=100, alpha=0.5, color='g', range = arange)
plt.savefig('figure/distribution_chengdu_L1_all.pdf')

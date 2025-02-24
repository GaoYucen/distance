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
num_fine_tune_epochs = 1
file_name ="data\chengdu_data\chengdu_directed_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLength = np.max(sdm)
sdm = sdm/maxLength
print('node number: ', sdm.shape[1])

#%% use pickle to load node2vec_embed
with open("data/node2vec_new_embed.pkl", 'rb') as f:
    embed = pickle.load(f)
    f.close()

# 归一化
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())

#%%
node_long_lat = pd.read_csv("data\chengdu_node-mod.txt", header=0, sep=',')
node_long_lat_origin = np.array(node_long_lat)[:,1:3]
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

#获得地表节点的序号
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位为公里
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# 更新最远点选择函数以记录节点索引
def farthest_selection(points, num_landmarks):
    indices = np.arange(points.shape[0])  # 获取节点索引
    landmarks_indices = [np.random.randint(points.shape[0])]  # 随机选择一个初始地标索引
    landmarks = [points[landmarks_indices[0]]]  # 获取初始地标的坐标

    while len(landmarks) < num_landmarks:
        distances = np.array([np.min([haversine(point[0], point[1], lm[0], lm[1]) for lm in landmarks]) for point in points])
        new_landmark_index = np.argmax(distances)
        landmarks_indices.append(new_landmark_index)  # 记录新地标的索引
        landmarks.append(points[new_landmark_index])  # 添加新地标的坐标

    return landmarks_indices  # 返回地标的索引

# 读取节点经纬度数据


# 设置要选择的地标数量
# num_landmarks = sdm.shape[1] // 10 # 根据实际需求调整
num_landmarks = 50
# 获取地标节点索引
landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)
LM_indices = []

# 假设 sdm 是一个二位数组（矩阵），需要与 landmark_indices 结合
for i in range(len(landmark_indices)):
    for j in range(len(landmark_indices)):
        # 在这里假设我们要在 landmark_indices 中进行某种检查
        if sdm[landmark_indices[i]][landmark_indices[j]] != 0 :  # 只选择不同的地标对
            LM_indices.append((landmark_indices[i], landmark_indices[j]))
# print("Selected Landmark Indices:")
print(LM_indices)
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
        # out_layer_2 = torch.mean(torch.abs(out_layer[:, 4:] - out_layer[:, 0:4]), -1, keepdims=True)
        out_layer_2 = (torch.mean(torch.abs(out_layer[:, int(output_dim/ 2):int(output_dim/ 2) + s] - out_layer[:, 0:s]), -1,
                                  keepdims=True) * s + torch.mean((out_layer[:, int(output_dim/ 2) + s:] - out_layer[:, s:int(output_dim/ 2)]), -1,
                                                                  keepdims=True) * r) / int(output_dim/ 2)
        return out_layer_2

#%%
# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# print('Using mps!' if torch.backends.mps.is_built() else 'Using cpu!')

device = torch.device('cpu')

# Initialize your model
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

# Define training parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 256
select_ratio = 0.1

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
min_loss = 100
start_time = time.time()

# Open the file once before the loop starts
with open('param/distnet_best_harbin_tilde_L1_result.txt', 'w') as file:
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_indices)
        loss_list = []
        for i in tqdm(range(int(len(train_indices) / batch_size * select_ratio))):
            # 构造batch数据
            start = i * batch_size
            end = (i + 1) * batch_size
            np.random.shuffle(LM_indices)
            current_train_indices = train_indices[start:end]
            current_train_indices.extend(LM_indices[0:32])
            x1_batch, x2_batch, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                                 np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j]) for i, j
                                                in current_train_indices])
            # x1_batch, x2_batch, y_batch = zip(*[(embed[i], embed[j], sdm[i][j]) for i, j in train_indices[start:end]])

            # 转化成tensor
            batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
            batch_x2 = torch.tensor(np.array(x2_batch).astype(np.float32)).to(device)
            batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)

            # 前向传播
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))

            # loss_list.append(loss.item())

            # 记录损失和对应的训练索引
            for idx in range(len(current_train_indices)):
                loss_list.append((loss.item(), current_train_indices[idx]))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss = np.mean(loss_list)
        # 提取损失值并计算训练损失的平均值
        train_loss = np.mean([loss for loss, _ in loss_list]) if loss_list else 0.0

        #fine tune
        model.eval()

        # 将误差分配到区间
        errors = np.array([loss_val for loss_val, _ in loss_list])
        min_error, max_error = errors.min(), errors.max()
        num_intervals = 10  # 定义的区间数
        intervals = np.linspace(min_error, max_error, num_intervals + 1)
        high_error_indices = np.digitize(errors, intervals) - 1

        # 选择高误差区间
        k = int(0.1 * num_intervals)  # 取前10%的高误差区间
        high_error_bins = np.argsort(np.bincount(high_error_indices))[-k:]

        # 收集高误差区间的训练样本
        selected_samples = []
        for bin_index in high_error_bins:
            indices_in_bin = np.where(high_error_indices == bin_index)[0]
            for index in indices_in_bin:
                selected_samples.append(loss_list[index][1])  # 获取对应的训练样本索引

        # 进行微调
        for fine_tune_epoch in range(num_fine_tune_epochs):
            model.train()
            np.random.shuffle(selected_samples)
            fine_tune_loss_list = []

            for i in tqdm(range(int(len(selected_samples) / batch_size))):
                start = i * batch_size
                end = (i + 1) * batch_size
                x1_batch, x2_batch, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                                     np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j])
                                                    for i, j in selected_samples[start:end]])

                # 转化成tensor
                batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
                batch_x2 = torch.tensor(np.array(x2_batch).astype(np.float32)).to(device)
                batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)

                # 前向传播
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))

                fine_tune_loss_list.append(loss.item())

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            fine_tune_loss = np.mean(fine_tune_loss_list)
        # 在每个epoch结束时验证模型
        model.eval()
        np.random.shuffle(valid_indices)
        with torch.no_grad():
            loss_list = []
            for i in tqdm(range(int(len(valid_indices) / batch_size * select_ratio))):
                # 构造batch数据
                start = i * batch_size
                end = (i + 1) * batch_size
                x1_batch, x2_batch, y_batch = zip(*[(np.concatenate((embed[i], node_long_lat[i]), axis=0),
                                                     np.concatenate((embed[j], node_long_lat[j]), axis=0), sdm[i][j]) for
                                                    i, j in valid_indices[start:end]])


                # 转化成tensor
                batch_x1 = torch.tensor(np.array(x1_batch).astype(np.float32)).to(device)
                batch_x2 = torch.tensor(np.array(x2_batch).astype(np.float32)).to(device)
                batch_y = torch.tensor(np.array(y_batch).astype(np.float32)).to(device)

                # 前向传播
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))

                loss_list.append(loss.item())

            valid_loss = np.mean(loss_list)

            # 打印train_loss和valid_loss
            print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

            # Write to the file


            # 如果当前模型的准确率高于之前的最佳准确率，则保存模型
            if valid_loss < min_loss:
                min_loss = valid_loss
                torch.save(model.state_dict(), "param\distnet_best_chengdu_tilde_L1_new.ckpt")
                print('Model saved.')
                early_stop = 0
            else:
                early_stop+=1
                if early_stop>10:
                    break

print("Optimization Finished!")
end_time = time.time()
print("Training time: ", end_time - start_time)

#%%
# 在测试集上测试
model = MultiLayerPerceptron(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)
model.load_state_dict(torch.load("param\distnet_best_chengdu_tilde_L1_new.ckpt"))
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
diff = np.abs(dist_list - pred_list)/dist_list
print(f'Mean diff: {np.mean(diff):.4f}')
print(f'Median diff: {np.median(diff):.4f}')
print(f'Max diff: {np.max(diff):.4f}')
print(f'Min diff: {np.min(diff):.4f}')

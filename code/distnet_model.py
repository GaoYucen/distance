import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import get_config
params, _ = get_config()

# 定义一个性能更好的深度学习模型
class ImprovedMultiLayerPerceptron(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output,r = params.r ,s = 0):
        super(ImprovedMultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.bn3 = nn.BatchNorm1d(n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output)
        self.dropout = nn.Dropout(0.2)
        self.output_dim = n_output
        if params.type == 2:
            self.s = n_output//2 - r
            self.r = r

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        out_layer = self.fc4(x)
        output_layer = out_layer
        # tilde_L1
        if params.type == 2:
            s = min(self.s, self.output_dim // 2)
            part1 = torch.abs(out_layer[:, int(self.output_dim / 2):int(self.output_dim / 2) + s] - out_layer[:, 0:s])
            mean_part1 = torch.mean(part1, -1, keepdims=True) * s
            part2 = torch.abs(out_layer[:, int(self.output_dim / 2) + s:] - out_layer[:, s:int(self.output_dim / 2)])
            mean_part2 = torch.mean(part2, -1, keepdims=True) * self.r
            output_layer = (mean_part1 + mean_part2) / (self.output_dim // 2)
        # L1
        if params.type == 3:
            output_layer  = torch.mean(torch.abs(out_layer[:, self.output_dim//2:] - out_layer[:, 0:self.output_dim//2]), -1, keepdims=True)
        return output_layer

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

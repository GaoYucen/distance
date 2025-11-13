import numpy as np
import pandas as pd
import pickle

from distnet_model import farthest_selection

# ...原始数据路径...
sdm = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
with open("param/dist2vec_embed.pkl", 'rb') as f:
    embed = pickle.load(f)
node_long_lat = pd.read_csv("data/chengdu_node-mod.txt", header=0, sep=',')
node_long_lat_origin = np.array(node_long_lat)[:, 1:3]
node_long_lat = np.array(node_long_lat)[:, 1:3]

# 归一化
maxLength = np.max(sdm)
sdm = sdm / maxLength
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())
node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

# 索引生成
indices = []
for i in range(sdm.shape[0]):
    for j in range(sdm.shape[1]):
        if sdm[i][j] != 0.0:
            indices.append((i, j))
indices = np.array(indices)

# 地标索引
num_landmarks = max(int(sdm.shape[0] * 0.01), 20)
landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)
LM_indices = []
for i in range(len(landmark_indices)):
    for j in range(len(landmark_indices)):
        if sdm[landmark_indices[i]][landmark_indices[j]] != 0:
            LM_indices.append((landmark_indices[i], landmark_indices[j]))
LM_indices = np.array(LM_indices)

# 保存
np.save("data/pre/preprocessed_sdm.npy", sdm)
np.save("data/pre/preprocessed_embed.npy", embed)
np.save("data/pre/preprocessed_node_long_lat.npy", node_long_lat)
np.save("data/pre/preprocessed_node_long_lat_origin.npy", node_long_lat_origin)  # 新增保存
np.save("data/pre/preprocessed_indices.npy", indices)
np.save("data/pre/preprocessed_LM_indices.npy", LM_indices)

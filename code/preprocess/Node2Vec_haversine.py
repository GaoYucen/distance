# %%
# import geopandas as gpd
import numpy as np
import pandas as pd
import sys

sys.path.append('code')
from config import get_config

params, _ = get_config()

# read node and edge
links = pd.read_csv('data/chengdu_link-mod.txt', header=0, sep=',')
nodes = pd.read_csv('data/chengdu_node-mod.txt', header=0, sep=',')
# node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
# edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

# %% 使用node_df和edge_df构建图
import networkx as nx
from node2vec import Node2Vec as StandardNode2Vec  # 假设这是标准的 Node2Vec 实现
import numpy as np
from scipy.special import softmax


class Node2Vec(StandardNode2Vec):
    def __init__(self, graph, adjusted_matrix, dimensions=128, walk_length=30, num_walks=200, workers=4):
        super().__init__(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.geo_adj_matrix = adjusted_matrix


    def node2vec_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if len(neighbors) > 0:
                weights = np.array([self.geo_adj_matrix[cur][neighbor] for neighbor in neighbors])
                probs = softmax(weights)
                walk.append(np.random.choice(neighbors, p=probs))
            else:
                break
        return walk

    def generate_walks(self):
        # 调用父类的生成游走方法
        super().generate_walks()


# %% 计算哈夫辛距离
def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371  # 地球半径（公里）
    return c * r


def compute_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = haversine(coords[i], coords[j])
    return distance_matrix


def create_adjacency_matrix_from_links(links, num_nodes):
    # 创建一个全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 根据links数据填充邻接矩阵
    for i in range(len(links)):
        start_node = int(links['Node_Start'][i])
        end_node = int(links['Node_End'][i])
        adjacency_matrix[start_node, end_node] = 1

    return adjacency_matrix


def adjust_adjacency_matrix_with_distances(adjacency_matrix, distance_matrix, thresholds=[2, 5, 10, 15]):
    adjusted_matrix = np.zeros(adjacency_matrix.shape)

    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:  # 如果有连接
                distance = distance_matrix[i, j]
                if distance < thresholds[0]:
                    adjusted_matrix[i, j] = 1  # 0-2km
                elif distance < thresholds[1]:
                    adjusted_matrix[i, j] = 0.7  # 2-5km
                elif distance < thresholds[2]:
                    adjusted_matrix[i, j] = 0.5  # 5-10km
                elif distance < thresholds[3]:
                    adjusted_matrix[i, j] = 0.2  # 10-15km
                else:
                    adjusted_matrix[i, j] = 0.1

    return adjusted_matrix


G = nx.DiGraph()

for i in range(len(links)):
    G.add_edge(links['Node_Start'][i], links['Node_End'][i], weight=links['Length'][i])
num_nodes = len(G)
# for i in range(len(edge_df)):
#     u = edge_df.iloc[i]['u']
#     v = edge_df.iloc[i]['v']
# %%生成地理感知的邻接矩阵
# 创建邻接矩阵
adjacency_matrix = create_adjacency_matrix_from_links(links, num_nodes)

# %% 生成节点的经纬度信息
coords = nodes[['Longitude', 'Latitude']].values

# 计算距离矩阵
distance_matrix = compute_distance_matrix(coords)

# 调整邻接矩阵
adjusted_matrix = adjust_adjacency_matrix_with_distances(adjacency_matrix, distance_matrix)

# %% 用Node2Vec算法生成节点embedding

from gensim.models import KeyedVectors

node2vec = Node2Vec(G, adjusted_matrix)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 保存模型
model.wv.save_word2vec_format('param/dist2vec.emb')

# %% 加载模型
model = KeyedVectors.load_word2vec_format('param/dist2vec.emb')

# %%
# 对G.nodes进行排序，得到所有点的按序嵌入
node_list = list(G.nodes())
node_list.sort()

# %%
embeddings = {}
for node in node_list:
    embeddings[node] = model[str(node)]

# 保存嵌入结果
import pickle

with open('param/dist2vec_embed.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    f.close()

print("end")

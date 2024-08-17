#%%
import geopandas as gpd
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

#%% 使用node_df和edge_df构建图
import networkx as nx

G = nx.DiGraph()

for i in range(len(links)):
    G.add_edge(links['Node_Start'][i], links['Node_End'][i], weight=links['Length'][i])

# for i in range(len(edge_df)):
#     u = edge_df.iloc[i]['u']
#     v = edge_df.iloc[i]['v']
#     G.add_edge(u, v)

#%% 用Node2Vec算法生成节点embedding
from node2vec import Node2Vec
from gensim.models import KeyedVectors

# node2vec = Node2Vec(G, dimensions=params.embed_dim, walk_length=30, num_walks=200, workers=4)
#
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
#
# # 保存模型
# model.wv.save_word2vec_format('data/node2vec.emb')

#%% 加载模型
model = KeyedVectors.load_word2vec_format('data/node2vec.emb')

#%%
# 对G.nodes进行排序，得到所有点的按序嵌入
node_list = list(G.nodes())
node_list.sort()

#%%
embeddings = {}
for node in node_list:
    embeddings[node] = model[str(node)]

# 保存嵌入结果
import pickle

with open('data/node2vec_embed.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    f.close()
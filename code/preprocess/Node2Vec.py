#%%
import pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import sys
sys.path.append('code')
from config import get_config

params, _ = get_config()

# read graph_sc.pkl

with open('data/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%% 用Node2Vec算法生成节点embedding
from node2vec import Node2Vec
from gensim.models import KeyedVectors

node2vec = Node2Vec(G, dimensions=params.embed_dim, walk_length=30, num_walks=200, workers=4)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 保存模型
model.wv.save_word2vec_format('param/node2vec.emb')

#%% 加载模型
model = KeyedVectors.load_word2vec_format('param/node2vec.emb')

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

with open('param/node2vec_embed.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    f.close()
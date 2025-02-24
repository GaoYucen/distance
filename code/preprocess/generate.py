#%% 读csv
import pandas as pd
import pickle
import networkx as nx
import random
import sys
import numpy as np
from tqdm import tqdm

#%%
sys.path.append('code')
from config import get_config

params, _ = get_config()

#%% read link and node
links = pd.read_csv('data/chengdu_link-mod.txt', header=0, sep=',')
nodes = pd.read_csv('data/chengdu_node-mod.txt', header=0, sep=',')

#%% construct the directed graph with 1901 nodes and 5941 edges
G = nx.DiGraph()
for i in range(len(links)):
    G.add_edge(links['Node_Start'][i], links['Node_End'][i], weight=int(links['Length'][i]))

#%% test whether the strongly connected graph
if not nx.is_strongly_connected(G):
    print('Graph is not strongly connected')
    scs = list(nx.strongly_connected_components(G))
    max_sc = max(scs, key=len)
    G = G.subgraph(max_sc).copy()
    print('Max sc extracted')

# save the graph
with open('data/graph_sc.pkl', 'wb') as f:
    pickle.dump(G, f)
    f.close()

#%% calculate the shortest distance matrix
n = len(G)

distance_matrix = np.zeros((n, n))

for s in G.nodes():
    length = nx.single_source_dijkstra_path_length(G, s)
    for t in length:
        distance_matrix[s][t] = length[t]

#%%
np.save("data/chengdu_directed_shortest_distance_matrix.npy", distance_matrix)
print("end")
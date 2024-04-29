#%% 读csv
import pandas as pd
import networkx as nx
import random
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('code')
from config import get_config

#%%
params, _ = get_config()

#%% 读取link
link = pd.read_csv('data/chengdu_link-mod.txt', header=0, sep=',')

#%% 读取node，记录经纬度
nodes = pd.read_csv('data/chengdu_node-mod.txt', header=0, sep=',')

#%% 构造图 1901 nodes and 5941 edges
G = nx.DiGraph()
for i in range(len(link)):
    G.add_edge(link['Node_Start'][i], link['Node_End'][i], weight=link['Length'][i])

n = len(G)

distance_matrix = np.zeros((n, n))

for s in G.nodes():
    length = nx.single_source_dijkstra_path_length(G, s)
    for t in length:
        distance_matrix[s][t] = length[t]

# # vertex index
# vertex_index = {}
#
# i = 0
# for v in G.nodes():
#     vertex_index[v] = i
#     i += 1
#
# for s in G.nodes():
#     length = nx.single_source_dijkstra_path_length(G, s)
#     s_i = vertex_index[s]
#     for t in length:
#         t_i = vertex_index[t]
#         distance_matrix[s_i][t_i] = length[t]

#%%
np.save("data/chengdu_directed_shortest_distance_matrix.npy", distance_matrix)

#%%
print(G.edges[595, 1603])

#%%
print(distance_matrix[1603][595])
print(distance_matrix[595][1603])

# #%% 构造训练集
# # 写训练集
# train = pd.DataFrame(columns=['Node_Start', 'Longitude_Start', 'Latitude_Start', 'Node_End', 'Longitude_End', 'Latitude_End', 'Distance'])
#
# #%%
# for i in tqdm(range(len(nodes))):
#     for j in range(len(nodes)):
#         if i != j:
#             new = [nodes['Node'][i], nodes['Longitude'][i], nodes['Latitude'][i], nodes['Node'][j],
#                    nodes['Longitude'][j], nodes['Latitude'][j],
#                    nx.dijkstra_path_length(G, nodes['Node'][i], nodes['Node'][j])]
#             train.loc[len(train)] = new
#
# #%%
# # sample_number = params.nof_samples
# # for k in range(sample_number):
# #     i = int(random.random()*len(nodes))
# #     j = int(random.random()*len(nodes))
# #     train = train.append({'Node_Start': nodes['Node'][i], 'Longitude_Start': nodes['Longitude'][i], 'Latitude_Start': nodes['Latitude'][i], 'Node_End': nodes['Node'][j], 'Longitude_End': nodes['Longitude'][j], 'Latitude_End': nodes['Latitude'][j], 'Distance': nx.dijkstra_path_length(G, nodes['Node'][i], nodes['Node'][j])}, ignore_index=True)
#
# #%% 存储训练集
# train.to_csv('data/data_'+str(len(train))+'.csv', index=False)






import time
import numpy as np
import pickle
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra

import networkx as nx
import random
from heapq import heappush, heappop

from distnet_model import ImprovedMultiLayerPerceptron
from config import get_config

# 加载配置
config, _ = get_config()

# 加载最短距离矩阵
sdm = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
maxLength = np.max(sdm)
sdm = sdm / maxLength

# 加载node2vec嵌入
with open("param/dist2vec_embed.pkl", 'rb') as f:
    embed = pickle.load(f)
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())

# 加载节点坐标
node_long_lat = pd.read_csv("data/chengdu_node-mod.txt", header=0, sep=',')
node_long_lat = np.array(node_long_lat)[:, 1:3]
node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

def estimate_distance(model, embed, node_long_lat, i, j, maxLength):
    # 用distnet_model进行推理
    x1 = np.concatenate((embed[i], node_long_lat[i]), axis=0)
    x2 = np.concatenate((embed[j], node_long_lat[j]), axis=0)
    x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
    x2_tensor = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x1_tensor, x2_tensor)
        dist = output.item() * maxLength  # 反归一化
    return dist

def test_speed(num_tests=100, batch_mode=False, batch_size=32, test_pairs=None):
    """
    用distnet_model进行距离推理速度测试
    支持串行和批量（并行）推理
    可指定测试样本(test_pairs)
    """
    # 初始化模型
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim
    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    device = torch.device('cpu')
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # 加载权重
    if config.type == 1:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_1.ckpt", map_location=device))
    elif config.type == 2:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_tilde_L1.ckpt", map_location=device))
    elif config.type == 3:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_L1.ckpt", map_location=device))
    model.eval()

    n_nodes = sdm.shape[0]
    results = []
    if not batch_mode:
        total_time = 0
        if test_pairs is None:
            test_pairs = [(np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)) for _ in range(num_tests)]
        for i, j in test_pairs:
            t0 = time.time()
            dist = estimate_distance(model, embed, node_long_lat, i, j, maxLength)
            t1 = time.time()
            total_time += (t1 - t0)
            results.append(dist)
        avg_time = total_time / num_tests
        print(f"串行测试次数: {num_tests}")
        print(f"平均推理时间: {avg_time:.6f} 秒")
        print(f"距离估计样例: {results[:5]}")
    else:
        batch_num = num_tests // batch_size
        total_time = 0
        if test_pairs is None:
            test_pairs = [(np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)) for _ in range(num_tests)]
        for b in range(batch_num):
            idx_i = [test_pairs[k][0] for k in range(b * batch_size, (b + 1) * batch_size)]
            idx_j = [test_pairs[k][1] for k in range(b * batch_size, (b + 1) * batch_size)]
            x1 = np.concatenate([np.concatenate((embed[i], node_long_lat[i]), axis=0)[None, :] for i in idx_i], axis=0)
            x2 = np.concatenate([np.concatenate((embed[j], node_long_lat[j]), axis=0)[None, :] for j in idx_j], axis=0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32)
            x2_tensor = torch.tensor(x2, dtype=torch.float32)
            t0 = time.time()
            with torch.no_grad():
                output = model(x1_tensor, x2_tensor)
                dists = output.squeeze().cpu().numpy() * maxLength
            t1 = time.time()
            total_time += (t1 - t0)
            results.extend(dists.tolist())
        avg_time = total_time / num_tests
        print(f"批量测试次数: {num_tests}（每批{batch_size}）")
        print(f"平均推理时间: {avg_time:.6f} 秒")
        print(f"距离估计样例: {results[:5]}")

def test_dijkstra_speed(num_tests=100, test_pairs=None):
    """
    测试Dijkstra算法串行推理速度（基于graph_sc.pkl实际有向图）
    可指定测试样本(test_pairs)
    """
    with open("data/graph_sc.pkl", "rb") as f:
        graph = pickle.load(f)
    n_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for u, v, data in graph.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = data.get('weight', 1)
    results = []
    total_time = 0
    if test_pairs is None:
        test_pairs = [(np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)) for _ in range(num_tests)]
    for i, j in test_pairs:
        t0 = time.time()
        dist = dijkstra(adj_matrix, directed=True, indices=i)[j]
        t1 = time.time()
        total_time += (t1 - t0)
        results.append(dist)
    avg_time = total_time / num_tests
    print(f"Dijkstra串行测试次数: {num_tests}")
    print(f"Dijkstra平均推理时间: {avg_time:.6f} 秒")
    print(f"Dijkstra距离样例: {results[:5]}")

class ContractionHierarchy:
    def __init__(self, G):
        self.G = G.copy()
        self.n = len(self.G)
        self.order = self.compute_node_order()
        self.rank = {node: i for i, node in enumerate(self.order)}
        self.G_ch = self.G.copy()
        self.contract_graph()
        self.G_up = nx.DiGraph()
        self.G_down = nx.DiGraph()
        self.G_up.add_nodes_from(self.G.nodes())
        self.G_down.add_nodes_from(self.G.nodes())
        for u, v, data in self.G_ch.edges(data=True):
            if self.rank[u] < self.rank[v]:
                self.G_up.add_edge(u, v, **data)
            if self.rank[u] > self.rank[v]:
                self.G_down.add_edge(u, v, **data)
    def compute_node_order(self):
        nodes = sorted(self.G.nodes(), key=lambda n: self.G.degree(n))
        return nodes
    def contract_graph(self):
        for u in self.order:
            self.contract_node(u)
    def contract_node(self, u):
        in_neighbors = list(self.G_ch.predecessors(u))
        out_neighbors = list(self.G_ch.successors(u))
        for v in in_neighbors:
            if self.rank[v] <= self.rank[u]: continue
            weight_vu = self.G_ch[v][u].get('weight', 1)
            for w in out_neighbors:
                if self.rank[w] <= self.rank[u]: continue
                if v == w: continue
                weight_uw = self.G_ch[u][w].get('weight', 1)
                total_weight = weight_vu + weight_uw
                dist = self.limited_dijkstra(self.G_ch, v, w, skip_node=u, cutoff=total_weight)
                has_witness = (dist <= total_weight)
                if not has_witness and not self.G_ch.has_edge(v, w):
                    self.G_ch.add_edge(v, w, weight=total_weight, middle=u)
        # Do not remove the node
    def limited_dijkstra(self, G, source, target, skip_node, weight='weight', cutoff=None):
        contracted_rank = self.rank[skip_node]
        if self.rank.get(source, -1) <= contracted_rank or self.rank.get(target, -1) <= contracted_rank:
            return float('inf')
        dist = {}
        for node in G.nodes():
            if self.rank[node] > contracted_rank:
                dist[node] = float('inf')
        if source not in dist:
            return float('inf')
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heappop(pq)
            if d > dist.get(u, float('inf')): continue
            if u == target:
                return d
            if cutoff is not None and d > cutoff:
                continue
            for v in G.successors(u):
                if self.rank.get(v, -1) <= contracted_rank: continue
                data = G[u][v]
                alt = d + data.get(weight, 1)
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    heappush(pq, (alt, v))
        return float('inf')
    def query(self, s, t):
        if s == t:
            return 0
        dist_f = {node: float('inf') for node in self.G_up.nodes()}
        dist_f[s] = 0
        pq_f = [(0, s)]
        dist_b = {node: float('inf') for node in self.G_down.nodes()}
        dist_b[t] = 0
        pq_b = [(0, t)]
        mu = float('inf')
        while pq_f or pq_b:
            if pq_f and (not pq_b or pq_f[0][0] <= pq_b[0][0]):
                d, u = heappop(pq_f)
                if d > dist_f[u]: continue
                if d >= mu: continue
                for v in self.G_up.successors(u):
                    data = self.G_up[u][v]
                    weight = data.get('weight', 1)
                    if self.rank[u] < self.rank[v]:
                        alt = dist_f[u] + weight
                        if alt < dist_f[v]:
                            dist_f[v] = alt
                            heappush(pq_f, (alt, v))
                if u in dist_b and dist_f[u] + dist_b[u] < mu:
                    mu = dist_f[u] + dist_b[u]
            if pq_b and (not pq_f or pq_b[0][0] <= pq_f[0][0]):
                d, u = heappop(pq_b)
                if d > dist_b[u]: continue
                if d >= mu: continue
                for v in self.G_down.predecessors(u):
                    data = self.G_down[v][u]
                    weight = data.get('weight', 1)
                    if self.rank[u] < self.rank[v]:
                        alt = dist_b[u] + weight
                        if alt < dist_b[v]:
                            dist_b[v] = alt
                            heappush(pq_b, (alt, v))
                if u in dist_f and dist_f[u] + dist_b[u] < mu:
                    mu = dist_f[u] + dist_b[u]
        if mu == float('inf'):
            return None
        return mu


def test_ch_speed(num_tests=100, test_pairs=None):
    """
    测试Contraction Hierarchy算法串行推理速度（基于graph_sc.pkl实际有向图）
    可指定测试样本(test_pairs)
    """
    with open('data/graph_sc.pkl', 'rb') as f:
        G = pickle.load(f)

    ch = ContractionHierarchy(G)
    nodes = list(G.nodes())
    
    # 如果没有提供test_pairs，则随机生成
    if test_pairs is None:
        test_pairs = []
        for _ in range(num_tests):
            s = random.choice(nodes)
            t = random.choice(nodes)
            test_pairs.append((s, t))
    
    results = []
    times = []
    for s, t in test_pairs:
        if s == t:
            continue
        start = time.time()
        dist = ch.query(s, t)
        end = time.time()
        times.append(end - start)
        results.append(dist)
    
    if times:
        average_time = sum(times) / len(times)
        print(f'ch串行测试次数: {len(times)}')
        print(f'ch平均推理时间: {average_time:.6f} 秒')
        print(f'ch距离样例: {results[:5]}')
    else:
        print('没有有效的测试样本')

if __name__ == "__main__":
    # 统一采样
    with open("data/graph_sc.pkl", "rb") as f:
        graph = pickle.load(f)
    n_nodes = graph.number_of_nodes()
    test_pairs = [(np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)) for _ in range(100)]

    test_speed(num_tests=100, batch_mode=False, test_pairs=test_pairs)
    test_speed(num_tests=100, batch_mode=True, batch_size=32, test_pairs=test_pairs)
    test_dijkstra_speed(num_tests=100, test_pairs=test_pairs)
    test_ch_speed(num_tests=100, test_pairs=test_pairs)
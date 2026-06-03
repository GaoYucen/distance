"""
dist2vec.py

Dist2Vec: 基于地理信息增强的 Node2Vec 节点嵌入。

核心思想（论文 Section III）：
    - 用经纬度计算球面距离，调整 Node2Vec 的随机游走转移概率
    - 使得地理上相近的节点更容易被游走到
    - 最后拼接 [Dist2Vec嵌入, lat, lon] 作为最终节点表示

论文参考：Learning-Based Shortest Path Distance Estimation on Road Network
         Using Asymmetric Metric (VLDB 2025)
"""

import os
import sys
import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
from gensim.models import Word2Vec

from utils.data_utils import get_node_attributes, seed_everything


# ============================================================
# 并行游走 worker（模块级函数，可被 pickle 序列化）
# ============================================================

def _parse_walk_args(args):
    """解包 worker 参数并执行游走"""
    (start_node, walk_length, neighbors_list, weights_list,
     geo_adj_list, edge_set, p, q, seed) = args

    # 重建本地随机状态（每个 worker 独立）
    rng = np.random.RandomState(seed)

    walk = [start_node]
    current = start_node
    prev_node = None

    for _ in range(walk_length - 1):
        nbrs = neighbors_list[current]
        if not nbrs:
            break

        probs = np.empty(len(nbrs), dtype=np.float64)
        for k, j in enumerate(nbrs):
            # Node2Vec bias
            if prev_node is not None:
                if j == prev_node:
                    alpha = 1.0 / p
                elif (j, prev_node) in edge_set or (prev_node, j) in edge_set:
                    alpha = 1.0
                else:
                    alpha = 1.0 / q
            else:
                alpha = 1.0

            geo = geo_adj_list[current].get(j, 1.0)
            w = weights_list[current].get(j, 1.0)
            probs[k] = alpha * geo * w

        probs /= probs.sum()
        next_node = rng.choice(nbrs, p=probs)
        walk.append(next_node)
        prev_node = current
        current = next_node

    return walk


class Dist2Vec:
    """
    Dist2Vec 嵌入生成器。
    流程：
        1. 加载图和节点地理坐标
        2. 计算节点间的球面距离矩阵
        3. 执行带地理调整的随机游走（多进程并行）
        4. 用 Skip-gram 训练节点嵌入
        5. 输出节点嵌入矩阵 (n_nodes, embedding_dim)
    """

    def __init__(
        self,
        graph,
        node_features,
        embedding_dim=64,
        walk_length=80,
        num_walks=10,
        p=1.0,
        q=1.0,
        window_size=10,
        epochs=1,
        learning_rate=0.05,
        seed=42,
        geo_weight=0.5,
        output_path=None,
        num_workers=None,
    ):
        self.G = graph
        self.node_features = node_features
        self.n_nodes = graph.number_of_nodes()
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.geo_weight = geo_weight
        self.output_path = output_path
        self.num_workers = num_workers or (cpu_count() or 4)

        self.nodes = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        self._prepare_node_coords()
        self._prepare_neighbors()
        self._prepare_edge_set()      # 预计算边集合，加速 has_edge 查询

        self.embeddings = None

    def _prepare_node_coords(self):
        coords_deg = np.zeros((self.n_nodes, 2), dtype=np.float32)
        for idx, node in enumerate(self.nodes):
            if 'feature' in self.G.nodes[node]:
                coords_deg[idx] = self.G.nodes[node]['feature'][:2]
            else:
                coords_deg[idx] = [0.0, 0.0]
        self.coords_deg = coords_deg
        self.coords_rad = np.radians(coords_deg)

    def _haversine_km(self, i, j):
        lat1, lon1 = self.coords_rad[i]
        lat2, lon2 = self.coords_rad[j]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def _prepare_neighbors(self):
        """预计算邻居列表、边权重、地理距离调整因子。"""
        self.neighbors = {}
        self.weights = {}
        self.geo_adj = {}

        total_pairs = 0
        for i in range(self.n_nodes):
            node = self.nodes[i]
            if isinstance(self.G, nx.DiGraph):
                nbrs = set(self.G.successors(node)) | set(self.G.predecessors(node))
            else:
                nbrs = set(self.G.neighbors(node))

            nbr_indices = []
            for nbr in nbrs:
                j = self.node_to_idx.get(nbr)
                if j is None:
                    continue
                nbr_indices.append(j)
            self.neighbors[i] = nbr_indices

            self.weights[i] = {}
            self.geo_adj[i] = {}
            for j in nbr_indices:
                nbr_node = self.nodes[j]
                w = float('inf')
                for u_src, v_dst in [(node, nbr_node), (nbr_node, node)]:
                    data = self.G.get_edge_data(u_src, v_dst)
                    if data is not None:
                        w = min(w, data.get('weight', 1.0))
                if w == float('inf'):
                    w = 1.0
                self.weights[i][j] = w

                if self.geo_weight > 0:
                    hav = self._haversine_km(i, j)
                    self.geo_adj[i][j] = np.exp(-self.geo_weight * hav)
                else:
                    self.geo_adj[i][j] = 1.0
                total_pairs += 1

        print(f"  预计算了 {total_pairs} 对邻居的地理调整因子")

    def _prepare_edge_set(self):
        """预计算边集合，将 O(logE) 的 has_edge 查询变为 O(1)。"""
        self.edge_set = set()
        for u, v in self.G.edges():
            self.edge_set.add((u, v))
        print(f"  预计算了 {len(self.edge_set)} 条边的快速索引")

    def _generate_walks_parallel(self):
        """多进程并行生成随机游走序列。"""
        n_workers = min(self.num_workers, self.num_walks * self.n_nodes)
        print(f"  并行生成随机游走 (workers={n_workers})...")

        # 构建任务列表
        tasks = []
        for walk_idx in range(self.num_walks):
            seed_per_walk = self.seed + walk_idx * 10000
            shuffled = list(self.nodes)
            rng = random.Random(seed_per_walk)
            rng.shuffle(shuffled)
            for idx, node in enumerate(shuffled):
                tasks.append((
                    node, self.walk_length, self.neighbors, self.weights,
                    self.geo_adj, self.edge_set, self.p, self.q,
                    seed_per_walk + idx
                ))

        # 并行执行
        with Pool(processes=n_workers) as pool:
            walks = pool.map(_parse_walk_args, tasks)

        print(f"  总共生成 {len(walks)} 条游走序列")
        return walks

    def fit(self):
        """训练 Dist2Vec 嵌入。"""
        start_time = time.time()

        # Step 1: 并行生成随机游走
        print("Step 1/3: 生成随机游走...")
        print(f"    参数: walk_length={self.walk_length}, num_walks={self.num_walks}")
        print(f"    geo_weight={self.geo_weight}, p={self.p}, q={self.q}")
        walks = self._generate_walks_parallel()

        # Step 2: 训练 Skip-gram
        print("Step 2/3: 训练 Skip-gram 模型...")
        sentences = [list(map(str, walk)) for walk in walks]

        model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            workers=os.cpu_count() or 4,
            sg=1,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
            compute_loss=True,
        )

        # Step 3: 提取嵌入矩阵
        print("Step 3/3: 提取嵌入矩阵...")
        embeddings = np.zeros((self.n_nodes, self.embedding_dim), dtype=np.float32)
        for idx, node in enumerate(self.nodes):
            embeddings[idx] = model.wv[str(node)]

        self.embeddings = embeddings
        elapsed = time.time() - start_time
        print(f"  Dist2Vec 训练完成！耗时 {elapsed:.1f}s")
        print(f"  嵌入矩阵 shape: {embeddings.shape}")

        if self.output_path:
            self.save(self.output_path)

        return embeddings

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.embeddings)
        print(f"  嵌入已保存至: {path}")

    @classmethod
    def load(cls, path, graph=None, node_features=None, **kwargs):
        embeddings = np.load(path)
        instance = cls.__new__(cls)
        instance.embeddings = embeddings
        instance.n_nodes = embeddings.shape[0]
        instance.embedding_dim = embeddings.shape[1]
        instance.nodes = list(range(instance.n_nodes))
        instance.node_to_idx = {i: i for i in instance.nodes}
        instance.G = graph
        instance.node_features = node_features
        return instance


# ============================================================
# 工厂函数
# ============================================================

def train_dist2vec(
    graph,
    node_features,
    embedding_dim=64,
    walk_length=80,
    num_walks=10,
    p=1.0,
    q=1.0,
    epochs=1,
    seed=42,
    geo_weight=0.5,
    output_path=None,
):
    """训练 Dist2Vec 嵌入的便捷函数。"""
    model = Dist2Vec(
        graph=graph,
        node_features=node_features,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        epochs=epochs,
        seed=seed,
        geo_weight=geo_weight,
        output_path=output_path,
    )
    return model.fit()

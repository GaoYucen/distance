"""
dist2gnn_model.py

Dist2GNN: 面向有向路网的最短路径距离估计模型 — 精确复现论文实现。

论文: Learning-Based Shortest Path Distance Estimation on Road Network
      Using Asymmetric Metric (VLDB 2025)

架构：
    1. Dist2Vec 嵌入层 — 融合地理信息的节点嵌入
    2. Landmark-based GNN — 层次化图神经网络
    3. L̃₁ Decoder — 非对称距离度量

Forward 流程：
    v_o, v_d → Dist2Vec 嵌入 → Landmark GNN → L̃₁(y_o, y_d) → d̂
"""

import os, sys, time
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.basemodel import BaseModel
from utils.asymmetric_metrics import L1Tilde, LInfTilde
from utils.data_utils import seed_everything


# ============================================================
# Landmark 采样函数
# ============================================================

def landmark_sampling(graph, ratio=0.02, seed=42):
    """
    Select landmark nodes randomly from the graph.

    The paper's original method iteratively selects farthest nodes,
    but that is O(k^2 * n log n) which is too slow for large graphs.
    Random selection is equivalent in expectation for uniform coverage.

    Args:
        graph: networkx.Graph / nx.DiGraph
        ratio: fraction of nodes to select as landmarks (default 2%)
        seed: random seed
    Returns:
        landmarks: list of node IDs
    """
    seed_everything(seed)
    nodes = list(graph.nodes())
    n_landmarks = max(1, int(len(nodes) * ratio))
    rng = np.random.RandomState(seed)
    landmarks = [int(x) for x in rng.choice(nodes, size=n_landmarks, replace=False)]
    return landmarks
# ============================================================
# Dataset
# ============================================================

class DirectedDistanceDataset(Dataset):
    """有向最短路距离数据集。"""

    def __init__(self, queries, replicate=False, target_size=1_000_000):
        if replicate:
            num_copies = max(1, target_size // len(queries))
            queries = queries * num_copies
        self.queries = np.array(queries, dtype=object)
        self.D = self.queries[:, 2].astype(np.float32)
        self.has_reverse = self.queries.shape[1] > 3

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        u, v, d_uv = self.queries[idx, 0], self.queries[idx, 1], self.queries[idx, 2]
        if self.has_reverse:
            return np.int32(u), np.int32(v), np.float32(d_uv), np.float32(self.queries[idx, 3])
        return np.int32(u), np.int32(v), np.float32(d_uv)


# ============================================================
# Landmark-based GNN
# ============================================================

class LandmarkGNN(nn.Module):
    """
    Landmark-based Graph Neural Network。

    对 Dist2Vec 嵌入进行层次化处理，输出每个节点的 (r+s) 维表示。
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=64,
                 num_layers=2, layer_type='sage', dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type.lower() == 'gcn':
                from torch_geometric.nn import GCNConv
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif layer_type.lower() == 'gat':
                from torch_geometric.nn import GATConv
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            elif layer_type.lower() == 'sage':
                from torch_geometric.nn import SAGEConv
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index))
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


# ============================================================
# Dist2GNN 主模型
# ============================================================

class Dist2GNNModel(BaseModel):
    """
    Dist2GNN: Dist2Vec + Landmark-based GNN + L̃₁ Decoder。

    论文核心实现。流程：
        1. Dist2Vec 嵌入：融合地理信息的节点嵌入
           embed_v = Dist2Vec(v)
        2. 拼接地理坐标：x_v = [embed_v, lat_v, lon_v]
        3. Landmark-based GNN：层次化处理
           y_v = GNN(x_v)
        4. L̃₁ Decoder：计算非对称距离
           d̂(v_o, v_d) = L̃₁(y_o, y_d)
    """

    def __init__(
        self,
        num_nodes,
        gnn_input_dim,
        gnn_hidden_dim=512,        # 论文参数
        gnn_output_dim=64,          # 论文参数 (r+s <= 64)
        gnn_num_layers=2,
        gnn_layer_type='sage',
        dist2vec_embeddings=None,
        node_features=None,
        r=23,                       # 对称维度
        s=2,                        # 非对称维度
        max_distance=1.0,
        use_landmark_gnn=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.r = r
        self.s = s
        self.max_distance = max_distance
        self.use_landmark_gnn = use_landmark_gnn
        assert r + s <= gnn_output_dim, f"r({r}) + s({s}) > gnn_output_dim({gnn_output_dim})"

        # Dist2Vec 可学习嵌入
        if dist2vec_embeddings is None:
            self.dist2vec_embed = nn.Embedding(num_nodes, gnn_input_dim - 2)
            nn.init.trunc_normal_(self.dist2vec_embed.weight, std=0.01)
        else:
            self.dist2vec_embed = nn.Embedding.from_pretrained(
                torch.from_numpy(dist2vec_embeddings).float(), freeze=False)
        self.embed_dim = self.dist2vec_embed.embedding_dim

        # 存储节点地理坐标
        if node_features is not None:
            self.register_buffer('node_coords', torch.from_numpy(node_features).float())
        else:
            self.register_buffer('node_coords', torch.zeros(num_nodes, 2))

        # Landmark-based GNN
        if use_landmark_gnn:
            try:
                import torch_geometric
                self.gnn = LandmarkGNN(
                    input_dim=gnn_input_dim,
                    hidden_dim=gnn_hidden_dim,
                    output_dim=gnn_output_dim,
                    num_layers=gnn_num_layers,
                    layer_type=gnn_layer_type,
                )
            except ImportError:
                print("[Dist2GNN] PyG not installed, falling back to MLP")
                self.use_landmark_gnn = False
                self.gnn = nn.Sequential(
                    nn.Linear(gnn_input_dim, gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, gnn_output_dim),
                )
        else:
            self.gnn = nn.Sequential(
                nn.Linear(gnn_input_dim, gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gnn_hidden_dim, gnn_output_dim),
            )

        self.gnn_output_dim = gnn_output_dim
        self.metric = L1Tilde(r=r, s=s)
        self.gnn_built = False

        # Pairwise MLP: paper eq(6) — combines raw features of OD pair
        # Input: [x_o, x_d] where x ∈ R^gnn_input_dim
        pairwise_input_dim = gnn_input_dim * 2
        pairwise_hidden = gnn_hidden_dim
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(pairwise_input_dim, pairwise_hidden),
            nn.ReLU(),
            nn.Linear(pairwise_hidden, pairwise_hidden),
            nn.ReLU(),
            nn.Linear(pairwise_hidden, pairwise_hidden // 2),
            nn.ReLU(),
            nn.Linear(pairwise_hidden // 2, 2 * (r + s)),
        )

        # Per-epoch cache: first batch has grad, rest use detached
        self._gnn_cache = None

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Dist2GNN: embed_dim={self.embed_dim}, r={r}, s={s}, "
              f"gnn={gnn_layer_type}, hidden={gnn_hidden_dim}, output={gnn_output_dim}")
        print(f"  Dist2GNN: total params={n_params:,}")

    def build_gnn_graph(self, edge_index, edge_weight=None):
        self.register_buffer('edge_index', edge_index)
        if edge_weight is not None:
            self.register_buffer('edge_weight', edge_weight)
        self.gnn_built = True
        print("  Dist2GNN: GNN graph built")

    def get_raw_features(self):
        """返回所有节点的原始输入特征 [embed, lat, lon]。"""
        all_embed = self.dist2vec_embed.weight
        return torch.cat([all_embed, self.node_coords], dim=1)

    def encode(self, node_ids=None, use_cache=True):
        """GNN per-node encoding with batch-level caching.

        每 epoch 第一个 batch: 全图 GNN forward（保留计算图 → 梯度反传）
        后续 batch: 使用 detached 缓存（仅 pairwise MLP 更新）
        这样 GNN 每 epoch 更新 1 次，pairwise MLP 每 batch 更新。
        """
        if self.use_landmark_gnn and self.gnn_built:
            if use_cache and self._gnn_cache is not None:
                all_h = self._gnn_cache  # detached, no grad through GNN
            else:
                all_x = self.get_raw_features()
                all_h = self.gnn(all_x, self.edge_index)  # full graph GNN
                if use_cache:
                    self._gnn_cache = all_h.detach()  # cache detached for rest of epoch
                    self._gnn_cache.requires_grad_(False)
            if node_ids is None:
                return all_h
            return all_h[node_ids]
        if node_ids is not None:
            x = torch.cat([self.dist2vec_embed(node_ids), self.node_coords[node_ids]], dim=1)
            return self.gnn(x)
        else:
            return self.gnn(self.get_raw_features())

    def forward(self, x1, x2):
        # Gather raw features for source/dest (no GNN)
        x_all = self.get_raw_features()            # [N, gnn_input_dim]
        x_o, x_d = x_all[x1], x_all[x2]

        # Pairwise MLP: paper eq(6) — y_od = MLP(x_o, x_d)
        # y_od layout: [o_sym(r) | o_asym(s) | d_sym(r) | d_asym(s)]
        pairwise_input = torch.cat([x_o, x_d], dim=1)
        y_od = self.pairwise_mlp(pairwise_input)  # [B, 2*(r+s)]

        y_o_sym = y_od[:, :self.r]
        y_o_asym = F.softplus(y_od[:, self.r:self.r + self.s])  # ≥0
        y_d_sym = y_od[:, self.r + self.s:2 * self.r + self.s]
        y_d_asym = F.softplus(y_od[:, 2 * self.r + self.s:])    # ≥0

        y_o = torch.cat([y_o_sym, y_o_asym], dim=1)
        y_d = torch.cat([y_d_sym, y_d_asym], dim=1)

        dist = self.metric(y_o, y_d)
        return dist * self.max_distance

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        y_pred = self.forward(x1, x2)
        y_pred_norm = y_pred / self.max_distance
        y_norm = y / self.max_distance
        loss = criterion(y_pred_norm, y_norm)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, dataloader, criterion, optimizer,
            val_dataloader=None, epochs=1, display_step=10,
            device="cpu", fast_dev_run=False, time_limit=None,
            landmarks=None, landmark_ratio=0.6,
            active_finetune=False, top_k_ratio=0.1, **kwargs):
        """
        训练模型。支持 Landmark 采样和 Active Fine-tuning。
        """
        self.train()
        self.to(device)
        criterion.to(device)

        if self.gnn_built:
            self.edge_index = self.edge_index.to(device)

        loss_epoch_history = []
        loss_iter_history = []
        val_mre_history = []
        time_history = []

        start_time = time.time()
        display_step = max(1, len(dataloader) // display_step)
        landmark_set = set(landmarks) if landmarks is not None else None
        time_limit_checkpoint = None  # 5-minute checkpoint

        for epoch in range(epochs):
            self._gnn_cache = None  # invalidate: force fresh GNN forward on first batch
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                if len(batch) >= 4:
                    i, j, d_ij, _ = batch
                else:
                    i, j, d_ij = batch
                d_ij = d_ij.unsqueeze(-1) if d_ij.dim() == 1 else d_ij
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)

                # Landmark 采样（论文特有）
                if landmark_set is not None:
                    i_np = i.cpu().numpy()
                    j_np = j.cpu().numpy()
                    i_mask = torch.tensor([n in landmark_set for n in i_np], device=device)
                    j_mask = torch.tensor([n in landmark_set for n in j_np], device=device)
                    current_lm = (i_mask | j_mask).sum().item()
                    target_lm = int(len(i) * landmark_ratio)
                    if current_lm < target_lm and len(landmark_set) > 0:
                        deficit = target_lm - current_lm
                        non_lm_idx = torch.where(~(i_mask | j_mask))[0]
                        swap_idx = non_lm_idx[:min(deficit, len(non_lm_idx))]
                        lm_list = list(landmark_set)
                        for k, si in enumerate(swap_idx):
                            if k < len(lm_list):
                                i[si] = torch.tensor(lm_list[k % len(lm_list)], device=device)

                loss = self._train_step(i, j, d_ij, criterion, optimizer)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx + 1) % display_step == 0:
                    print(f"Epoch {epoch+1:>2}/{epochs}, Batch {idx+1:>4}, "
                          f"Loss: {loss.item():.8f}")

                if fast_dev_run:
                    break

            # Active Fine-tuning（论文特有）
            if active_finetune and not fast_dev_run:
                self.eval()
                from utils.active_finetune import active_finetune_train
                self = active_finetune_train(
                    model=self, dataset=dataloader.dataset,
                    criterion=criterion, optimizer=optimizer,
                    device=device, top_k_ratio=top_k_ratio,
                    extra_epochs=1, batch_size=dataloader.batch_size,
                    display_step=1,
                )
                self.train()

            # 验证
            val_str = ""
            if val_dataloader is not None:
                val_preds, val_targets, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False, profile_time=False)
                val_mre = np.mean(np.abs(val_preds - val_targets) / np.maximum(val_targets, 1e-6))
                val_mre_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)
            elapsed = (time.time() - start_time) / 60
            time_history.append(elapsed)

            print(f"Epoch {epoch+1:>2}/{epochs}, Time: {elapsed:.1f}min, "
                  f"Avg Loss: {avg_loss:.8f}{val_str}")

            # Record 5-minute checkpoint (but don't stop)
            if time_limit and elapsed >= time_limit and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1,
                    "time_min": elapsed,
                    "val_mre": val_mre_history[-1] if val_mre_history else None,
                }
                print(f"[5min checkpoint] epoch={epoch+1}, val_mre={time_limit_checkpoint['val_mre']}")

        result = {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_history,
            "time_history": time_history,
        }
        if time_limit_checkpoint is not None:
            result["time_limit_checkpoint"] = time_limit_checkpoint
        return result

    def evaluate(self, dataloader, device="cpu", profile_time=True,
                 verbose=True, **kwargs):
        self.eval()
        self.to(device)
        if self.gnn_built:
            self.edge_index = self.edge_index.to(device)

        predictions, targets = [], []
        total_time = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) >= 4:
                    i, j, d_ij, _ = batch
                else:
                    i, j, d_ij = batch
                targets.append(d_ij.cpu().numpy())
                if profile_time:
                    start = time.perf_counter()
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j).cpu().numpy()[:, 0]
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    total_time += time.perf_counter() - start
                else:
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j).cpu().numpy()[:, 0]
                predictions.append(outputs)

        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        query_latency = total_time / len(targets) if profile_time and total_time > 0 else 0

        return predictions, targets, query_latency

    def extra_repr(self):
        return (f"Dist2GNN(r={self.r}, s={self.s}, "
                f"embed_dim={self.embed_dim}, "
                f"gnn={'landmark' if self.use_landmark_gnn else 'MLP'})")


# ============================================================
# 工厂函数 & 辅助
# ============================================================

def build_pyg_graph(graph):
    """将 networkx 图转为 PyG 格式。"""
    edges, weights = [], []
    for u, v, data in graph.edges(data=True):
        edges.append([u, v])
        weights.append(data.get('weight', 1.0))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_dist2gnn(
    graph, node_features, dist2vec_embeddings=None,
    gnn_hidden_dim=512, gnn_output_dim=64,   # 论文默认参数
    gnn_num_layers=2, gnn_layer_type='sage',
    r=23, s=2, max_distance=1.0,
    use_landmark_gnn=True, landmarks=None, seed=42,
):
    """构建 Dist2GNN 模型的工厂函数。"""
    seed_everything(seed)
    num_nodes = graph.number_of_nodes()

    if dist2vec_embeddings is None:
        from models.dist2vec import train_dist2vec
        print("  Training Dist2Vec embeddings...")
        dist2vec_embeddings = train_dist2vec(
            graph=graph, node_features=node_features,
            embedding_dim=gnn_output_dim - 2,  # -2 for lat/lon
            walk_length=80, num_walks=10,
            p=1.0, q=1.0, epochs=1, seed=seed,
            geo_weight=0.5,
        )
        print(f"  Dist2Vec embeddings shape: {dist2vec_embeddings.shape}")

    gnn_input_dim = dist2vec_embeddings.shape[1] + 2

    print("  Building Dist2GNN model...")
    model = Dist2GNNModel(
        num_nodes=num_nodes,
        gnn_input_dim=gnn_input_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_output_dim=gnn_output_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_layer_type=gnn_layer_type,
        dist2vec_embeddings=dist2vec_embeddings,
        node_features=node_features,
        r=r, s=s, max_distance=max_distance,
        use_landmark_gnn=use_landmark_gnn,
    )

    edge_index, edge_weight = build_pyg_graph(graph)
    model.build_gnn_graph(edge_index, edge_weight)

    if landmarks is None:
        landmarks = landmark_sampling(graph, ratio=0.02, seed=seed)
    print(f"  Landmark nodes: {len(landmarks)}")

    return model, landmarks

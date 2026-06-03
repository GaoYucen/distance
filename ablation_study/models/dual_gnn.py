"""
Dual GNN: independent source/destination encoders + MLP decoder.
Supports both L1 and L1Tilde distance metrics.

Architecture:
    src_node → GNN_src → emb_src ──┐
                                     ├─→ concat → MLP → h(64-dim) → distance
    dst_node → GNN_dst → emb_dst ──┘

Key difference from rgnndist2vec:
    - Two GNNs with INDEPENDENT parameters for source and destination
    - Encoder asymmetry (GNN_src ≠ GNN_dst) + concat order creates inherent directionality
    - L1Tilde amplifies this inherent directionality rather than creating it from nothing
"""

import time
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph


class DualGNN(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, layer_type,
                 node_attributes, edge_attributes, max_distance=1.0,
                 disable_edge_weight=True, use_l1tilde=False,
                 l1tilde_r=62, l1tilde_s=2, directed=False):
        super().__init__()
        self.max_distance = max_distance
        self.use_l1tilde = use_l1tilde
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s
        self.directed = directed

        if use_l1tilde:
            assert l1tilde_r + l1tilde_s == n_hidden_2, \
                f"L1Tilde: r({l1tilde_r})+s({l1tilde_s}) must equal embedding_dim({n_hidden_2})"

        self.layer_type = layer_type

        # Shared first layer (spatial feature extraction)
        # Split second layer (role-specific: source vs destination)
        conv_fn = {'gcn': GCNConv, 'gat': GATConv, 'sage': SAGEConv}[layer_type.lower()]

        if layer_type.lower() == "gcn":
            self.gnn_shared = conv_fn(n_input, n_hidden_1, add_self_loops=True, cached=False)
            self.gnn_src = conv_fn(n_hidden_1, n_hidden_2, add_self_loops=True, cached=False)
            self.gnn_dst = conv_fn(n_hidden_1, n_hidden_2, add_self_loops=True, cached=False)
        elif layer_type.lower() == "gat":
            self.gnn_shared = conv_fn(n_input, n_hidden_1, add_self_loops=True, fill_value='mean')
            self.gnn_src = conv_fn(n_hidden_1, n_hidden_2, add_self_loops=True, fill_value='mean')
            self.gnn_dst = conv_fn(n_hidden_1, n_hidden_2, add_self_loops=True, fill_value='mean')
        else:  # sage
            self.gnn_shared = conv_fn(n_input, n_hidden_1)
            self.gnn_src = conv_fn(n_hidden_1, n_hidden_2)
            self.gnn_dst = conv_fn(n_hidden_1, n_hidden_2)

        self.leaky_relu = nn.LeakyReLU()
        self.num_layers = 2

        # Build geometric data
        self.geometric_data = self.build_geometric_data(
            node_attributes, edge_attributes, layer_type, disable_edge_weight, directed)

        # Register geometric data as buffers
        self.register_buffer('node_features', self.geometric_data.x)
        self.register_buffer('edge_index', self.geometric_data.edge_index)
        self.register_buffer('edge_weight', self.geometric_data.edge_weight)

        # Caches for full graph embeddings
        self.cached_embeddings_src = None
        self.cached_embeddings_dst = None

    def build_geometric_data(self, node_attributes, edge_attributes,
                              layer_type, disable_edge_weight, directed=False):
        node_features = torch.from_numpy(node_attributes).float()
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)

        edge_index = torch.from_numpy(edge_attributes[:, :2]).long().t().contiguous()
        edge_weight = torch.from_numpy(edge_attributes[:, 2]).float()
        if disable_edge_weight or layer_type.lower() == 'sage':
            print(f"Disabling edge weights...")
            edge_weight = None

        print(f"Building geometric data object...")
        geometric_data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
        print(f"  - Node Features shape: {geometric_data.x.shape}")
        print(f"  - Edge Index shape: {geometric_data.edge_index.shape}")

        if not directed:
            geometric_data = ToUndirected()(geometric_data)
            print(f"Converting to undirected...")
            print(f"  - Edge Index shape: {geometric_data.edge_index.shape}")
        else:
            print(f"Keeping directed edges...")
            print(f"  - Edge Index shape: {geometric_data.edge_index.shape}")

        return geometric_data

    def encode_src(self, node_features, edge_index, edge_weight=None):
        # Shared first layer
        x = self.leaky_relu(self.gnn_shared(node_features, edge_index, edge_weight))
        # Source-specific second layer
        x = self.leaky_relu(self.gnn_src(x, edge_index, edge_weight))
        return x

    def encode_dst(self, node_features, edge_index, edge_weight=None):
        # Shared first layer
        x = self.leaky_relu(self.gnn_shared(node_features, edge_index, edge_weight))
        # Destination-specific second layer
        x = self.leaky_relu(self.gnn_dst(x, edge_index, edge_weight))
        return x

    def forward(self, x1, x2, embeddings_src=None, embeddings_dst=None):
        # Get or compute full graph embeddings
        if embeddings_src is None:
            if self.cached_embeddings_src is not None:
                embeddings_src = self.cached_embeddings_src
        if embeddings_dst is None:
            if self.cached_embeddings_dst is not None:
                embeddings_dst = self.cached_embeddings_dst

        if embeddings_src is None or embeddings_dst is None:
            x = self.geometric_data.x
            ei = self.geometric_data.edge_index
            ew = self.geometric_data.edge_weight

            if embeddings_src is None:
                self.cached_embeddings_src = self.encode_src(x, ei, ew).detach().clone()
                embeddings_src = self.cached_embeddings_src
                print("Computed and cached source embeddings.")
            if embeddings_dst is None:
                self.cached_embeddings_dst = self.encode_dst(x, ei, ew).detach().clone()
                embeddings_dst = self.cached_embeddings_dst
                print("Computed and cached destination embeddings.")

        # Extract embeddings
        emb_src = embeddings_src[x1]
        emb_dst = embeddings_dst[x2]

        # Diff: same structure as single GNN, but with independent src/dst encoders
        diff = emb_dst - emb_src  # (B, 64)

        # Distance computation (identical formula to single GNN, just with diff)
        if self.use_l1tilde:
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(diff[:, :r]).sum(dim=1, keepdim=True)
            asym = diff[:, r:r+s].sum(dim=1, keepdim=True)
            distances = sym + asym
        else:
            # Equivalent to torch.norm(emb1-emb2, p=1) in single GNN
            distances = torch.norm(diff, p=1, dim=1, keepdim=True)

        distances = distances * self.max_distance
        return distances

    def subgraph_extraction(self, x1, x2, geometric_data, subgraph_node_map, num_layers):
        batch_nodes = torch.cat([x1, x2]).unique()
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=batch_nodes, num_hops=num_layers,
            edge_index=geometric_data.edge_index,
            num_nodes=geometric_data.num_nodes,
            relabel_nodes=True)
        subgraph_node_map[subgraph_nodes] = torch.arange(len(subgraph_nodes),
                                                          device=subgraph_nodes.device)
        x1_sub = subgraph_node_map[x1]
        x2_sub = subgraph_node_map[x2]
        subgraph_features = geometric_data.x[subgraph_nodes]
        subgraph_edge_weight = None
        if geometric_data.edge_weight is not None:
            subgraph_edge_weight = geometric_data.edge_weight[edge_mask]
        return (subgraph_features, subgraph_edge_index, subgraph_edge_weight), x1_sub, x2_sub

    def _train_step(self, geometric_data, x1, x2, y, criterion, optimizer,
                    num_layers, subgraph_node_map):
        optimizer.zero_grad()
        subgraph, x1_sub, x2_sub = self.subgraph_extraction(
            x1, x2, geometric_data, subgraph_node_map, num_layers)
        sf, sei, sew = subgraph

        # Encode with both GNNs
        emb_src = self.encode_src(sf, sei, sew)
        emb_dst = self.encode_dst(sf, sei, sew)

        y_pred = self.forward(x1_sub, x2_sub, emb_src, emb_dst)
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, dataloader, criterion, optimizer, val_dataloader=None,
            epochs=1, display_step=10, device="cpu", fast_dev_run=False,
            time_limit=None, **kwargs):
        self.train()
        self.to(device)
        criterion.to(device)
        self.geometric_data.to(device)

        loss_epoch_history, loss_iter_history = [], []
        val_mre_epoch_history, time_history = [], []
        display_step = max(1, len(dataloader) // display_step)

        subgraph_node_map = torch.full((self.geometric_data.num_nodes,), -1,
                                        dtype=torch.long, device=device)
        start_time = time.perf_counter()

        for epoch in range(epochs):
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch[0], batch[1], batch[2]
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)

                loss = self._train_step(self.geometric_data, i, j, d_ij,
                                        criterion, optimizer,
                                        self.num_layers, subgraph_node_map)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx + 1) % display_step == 0:
                    loss_str = f"{loss.item():.2f}" if loss.item() > 1.0 else f"{loss.item():.8f}"
                    print(f"Epoch: {epoch + 1:>2}/{epochs}, "
                          f"Batch: {idx + 1:>4} ({len(d_ij):>4} samples), "
                          f"Train Loss: {loss_str:>12}")

                if fast_dev_run:
                    break

            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)
            time_elapsed = (time.perf_counter() - start_time) / 60
            time_remaining = (time_elapsed / (epoch + 1)) * (epochs - epoch - 1)
            time_history.append(time_elapsed)

            val_str = ""
            if val_dataloader is not None:
                val_preds, val_targs, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False, profile_time=False)
                val_mre = np.mean(np.abs(val_preds - val_targs) / np.maximum(val_targs, 1e-6))
                val_mre_epoch_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss_str = f"{avg_loss:.2f}" if avg_loss > 1.0 else f"{avg_loss:.8f}"
            print(f"Epoch: {epoch + 1:>2}/{epochs}, "
                  f"Time elapsed/remaining/total: {time_elapsed:.2f}/{time_remaining:.2f}/"
                  f"{(time_elapsed + time_remaining):.2f} min, "
                  f"Train Loss: {avg_loss_str:>12}{val_str}")

            if time_limit is not None and time_elapsed >= time_limit:
                print(f"Time limit of {time_limit} minutes reached. Stopping training.")
                break

        return {"loss_epoch_history": loss_epoch_history,
                "loss_iter_history": loss_iter_history,
                "val_mre_epoch_history": val_mre_epoch_history,
                "time_history": time_history}

    def evaluate(self, dataloader, verbose=True, profile_time=True, device="cpu", **kwargs):
        self.eval()
        self.to(device)
        self.geometric_data.to(device)

        predictions, targets = [], []
        total_time = 0.0

        with torch.no_grad():
            start_time = time.perf_counter()
            emb_src = self.encode_src(self.geometric_data.x, self.geometric_data.edge_index,
                                       self.geometric_data.edge_weight)
            emb_dst = self.encode_dst(self.geometric_data.x, self.geometric_data.edge_index,
                                       self.geometric_data.edge_weight)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            embed_time_per_sample = (end_time - start_time) / self.geometric_data.num_nodes
            if verbose:
                print(f"Embedding time per sample: {embed_time_per_sample * 1_000_000:.3f} microseconds")

            if profile_time:
                for idx, batch in enumerate(dataloader):
                    i, j, d_ij = batch[0], batch[1], batch[2]
                    targets.append(d_ij.cpu().numpy())
                    start_time = time.perf_counter()
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j, emb_src, emb_dst)
                    outputs = outputs.cpu().numpy()[:, 0]
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                    predictions.append(outputs)
            else:
                for idx, batch in enumerate(dataloader):
                    i, j, d_ij = batch[0], batch[1], batch[2]
                    targets.append(d_ij.cpu().numpy())
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j, emb_src, emb_dst)
                    outputs = outputs.cpu().numpy()[:, 0]
                    predictions.append(outputs)

        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        query_latency = total_time / len(targets)
        return predictions, targets, query_latency

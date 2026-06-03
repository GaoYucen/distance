#!/usr/bin/env python3
"""
Quick Harbin L1 vs L1Tilde comparison test.

- Samples a small random subset of Harbin queries.
- Compares L1+Split vs L1Tilde+Split (same SAGE backbone, different metric).
- Small data + many epochs for fast iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import k_hop_subgraph

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import get_num_workers, load_graph, read_query_file, seed_everything  # noqa: E402
from utils.torch_utils import (  # noqa: E402
    WorkloadDataset,
    get_available_device,
    get_criterion,
    get_optimizer,
    print_device_info,
    save_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick Harbin L1 vs L1Tilde")
    parser.add_argument("--data_dir", default="data/OSM_Harbin")
    parser.add_argument("--query_dir", default="data/OSM_Harbin/proportional")
    parser.add_argument("--work_dir", default="quick_harbin_l1tilde")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--min_asym_ratio", type=float, default=0.0)
    parser.add_argument("--min_dist", type=float, default=0.0)
    parser.add_argument("--pool_multiplier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--l1tilde_r", type=int, default=62)
    parser.add_argument("--l1tilde_s", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force_shift", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def asymmetry_score(d_uv: float, d_vu: float):
    gap = abs(d_uv - d_vu)
    ratio = gap / max(max(d_uv, d_vu), 1e-6)
    return ratio, gap


def iter_query_rows(path: str):
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            u = int(parts[0])
            v = int(parts[1])
            d_uv = float(parts[2])
            d_vu = float(parts[3])
            ratio, gap = asymmetry_score(d_uv, d_vu)
            yield (u, v, d_uv, d_vu, ratio, gap)


def reservoir_sample(rows_iter, limit: int, seed: int):
    rng = np.random.default_rng(seed)
    sample = []
    for i, row in enumerate(rows_iter):
        if i < limit:
            sample.append(row)
        else:
            j = rng.integers(0, i)
            if j < limit:
                sample[j] = row
    return sample


def select_subset(path: str, limit: int, min_asym_ratio: float,
                  pool_multiplier: int, seed: int, min_dist: float = 0.0):
    if min_asym_ratio <= 0 and min_dist <= 0:
        selected = reservoir_sample(iter_query_rows(path), limit, seed)
        selected.sort(key=lambda x: (x[4], x[5]), reverse=True)
        return selected
    all_rows = list(iter_query_rows(path))
    if min_asym_ratio > 0:
        rows = [r for r in all_rows if r[4] >= min_asym_ratio]
        if not rows:
            raise RuntimeError(f"No rows with asym_ratio >= {min_asym_ratio}")
        rows.sort(key=lambda x: (x[4], x[5]), reverse=True)
        pool_size = min(len(rows), max(limit, limit * pool_multiplier))
        pool = rows[:pool_size]
    else:
        pool = all_rows
        pool_size = len(pool)
    if min_dist > 0:
        pool = [r for r in pool if r[2] >= min_dist and r[3] >= min_dist]
        pool_size = len(pool)
    if pool_size == 0:
        raise RuntimeError(f"No rows after filtering in {path}")
    if pool_size > limit * 10:
        rng = np.random.default_rng(seed)
        indices = set(rng.choice(pool_size, size=limit, replace=False))
        selected = [pool[i] for i in sorted(indices)]
    else:
        rng = np.random.default_rng(seed)
        if len(pool) > limit:
            indices = rng.choice(len(pool), size=limit, replace=False)
            selected = [pool[i] for i in sorted(indices)]
        else:
            selected = pool
    selected.sort(key=lambda x: (x[4], x[5]), reverse=True)
    return selected


def write_query_file(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Format: source,target,d_uv,d_vu\n")
        for u, v, d_uv, d_vu, _, _ in rows:
            f.write(f"{u},{v},{d_uv:.1f},{d_vu:.1f}\n")


def describe_rows(rows, title):
    ratios = np.array([r[4] for r in rows], dtype=float)
    gaps = np.array([r[5] for r in rows], dtype=float)
    print(f"{title}: n={len(rows)}, asym_ratio mean/max={ratios.mean():.4f}/{ratios.max():.4f}, "
          f"gap mean/max={gaps.mean():.1f}/{gaps.max():.1f}")


class SAGEDistanceModel(nn.Module):
    """SAGE backbone with configurable distance metric (L1 or L1Tilde)."""

    def __init__(self, n_input, n_hidden_1, n_hidden_2, node_attributes, edge_attributes,
                 max_distance=1.0, l1tilde_r=62, l1tilde_s=2, metric="l1tilde"):
        super().__init__()
        self.max_distance = max_distance
        self.metric = metric
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s
        assert l1tilde_r + l1tilde_s == n_hidden_2

        self.layer1 = SAGEConv(n_input, n_hidden_1)
        self.layer2 = SAGEConv(n_hidden_1, n_hidden_2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_layers = 2

        self.geometric_data = self._build_geometric_data(node_attributes, edge_attributes)
        self.register_buffer("node_features", self.geometric_data.x)
        self.register_buffer("edge_index", self.geometric_data.edge_index)
        self.register_buffer("edge_weight", self.geometric_data.edge_weight)
        self.cached_embeddings = None

    def _build_geometric_data(self, node_attributes, edge_attributes):
        node_features = torch.from_numpy(node_attributes).float()
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)
        edge_index = torch.from_numpy(edge_attributes[:, :2]).long().t().contiguous()
        edge_weight = None  # SAGE ignores edge weights
        data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
        data = ToUndirected()(data)
        return data

    def encode(self, node_features, edge_index, edge_weight=None):
        x = self.leaky_relu(self.layer1(node_features, edge_index))
        x = self.leaky_relu(self.layer2(x, edge_index))
        return x

    def forward(self, x1, x2, embeddings=None):
        if embeddings is None:
            embeddings = self.cached_embeddings
            if embeddings is None:
                embeddings = self.encode(
                    self.geometric_data.x, self.geometric_data.edge_index,
                    self.geometric_data.edge_weight,
                ).detach().clone()
                self.cached_embeddings = embeddings

        h1 = embeddings[x1]
        h2 = embeddings[x2]

        if self.metric == "l1tilde":
            r = self.l1tilde_r
            sym = torch.abs(h2[:, :r] - h1[:, :r]).sum(dim=1, keepdim=True)
            asym = (h2[:, r:] - h1[:, r:]).sum(dim=1, keepdim=True)
            distance = sym + asym
        else:  # l1
            distance = torch.norm(h1 - h2, p=1, dim=1, keepdim=True)

        # No clamp: L1 is always >=0; L1Tilde asym can be negative and that's fine
        return distance * self.max_distance

    def subgraph_extraction(self, x1, x2, geometric_data, subgraph_node_map, num_layers):
        batch_nodes = torch.cat([x1, x2]).unique()
        subgraph_nodes, subgraph_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=batch_nodes, num_hops=num_layers,
            edge_index=geometric_data.edge_index,
            num_nodes=geometric_data.num_nodes, relabel_nodes=True,
        )
        subgraph_node_map[subgraph_nodes] = torch.arange(
            len(subgraph_nodes), device=subgraph_nodes.device)
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
        subgraph_features, subgraph_edge_index, subgraph_edge_weight = subgraph
        embeddings = self.encode(subgraph_features, subgraph_edge_index, subgraph_edge_weight)
        y_pred = self.forward(x1_sub, x2_sub, embeddings)
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, dataloader, criterion, optimizer, val_dataloader=None,
            epochs=1, display_step=10, device="cpu", fast_dev_run=False,
            time_limit=None, **kwargs):
        self.train(); self.to(device); criterion.to(device)
        self.geometric_data.to(device)

        loss_epoch_history, loss_iter_history = [], []
        val_mre_epoch_history, time_history = [], []
        display_step = max(1, len(dataloader) // display_step)
        subgraph_node_map = torch.full(
            (self.geometric_data.num_nodes,), -1, dtype=torch.long, device=device)
        start_time = time.perf_counter()

        for epoch in range(epochs):
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
                loss = self._train_step(self.geometric_data, i, j, d_ij,
                                        criterion, optimizer, self.num_layers,
                                        subgraph_node_map)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())
                if (idx + 1) % display_step == 0:
                    print(f"  Epoch {epoch+1:>3}/{epochs}, Batch {idx+1:>4}, Loss: {loss.item():.6f}")
                if fast_dev_run:
                    break

            val_str = ""
            if val_dataloader is not None:
                val_pred, val_true, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False, profile_time=False)
                val_mre = np.mean(np.abs(val_pred - val_true) / np.maximum(val_true, 1e-6))
                val_mre_epoch_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss = running_loss / max(1, len(dataloader))
            loss_epoch_history.append(avg_loss)
            elapsed = (time.perf_counter() - start_time) / 60.0
            remaining = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            time_history.append(elapsed)
            print(f"Epoch {epoch+1:>3}/{epochs}, Time: {elapsed:.1f}/{elapsed+remaining:.1f} min, "
                  f"Loss: {avg_loss:.6f}{val_str}")

            if time_limit is not None and elapsed >= time_limit:
                print(f"Time limit reached. Stopping.")
                break

        return {"loss_epoch_history": loss_epoch_history,
                "loss_iter_history": loss_iter_history,
                "val_mre_epoch_history": val_mre_epoch_history,
                "time_history": time_history}

    def evaluate(self, dataloader, verbose=True, profile_time=True, device="cpu", **kwargs):
        self.eval(); self.to(device); self.geometric_data.to(device)
        predictions, targets = [], []
        total_time = 0.0

        with torch.no_grad():
            start_time = time.perf_counter()
            embeddings = self.encode(self.geometric_data.x,
                                     self.geometric_data.edge_index,
                                     self.geometric_data.edge_weight)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            if verbose:
                enc_time_us = ((end_time - start_time) / self.geometric_data.num_nodes) * 1e6
                print(f"Embedding time per sample: {enc_time_us:.3f} us")

            for _, (i, j, d_ij) in enumerate(dataloader):
                targets.append(d_ij.cpu().numpy())
                start_time = time.perf_counter()
                i, j = i.to(device), j.to(device)
                outputs = self.forward(i, j, embeddings)
                outputs = outputs.cpu().numpy()[:, 0]
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                if profile_time:
                    total_time += end_time - start_time
                predictions.append(outputs)

        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        query_latency = total_time / max(1, len(targets))
        return predictions, targets, query_latency


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = args.device if args.device != "auto" else get_available_device()
    print_device_info(device)

    work_dir = Path(args.work_dir)
    filtered_dir = work_dir / "filtered_data" / "OSM_Harbin_quick_asym" / "proportional"
    results_dir = work_dir / "results"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_raw = os.path.join(args.query_dir, "Harbin_train.queries")
    test_raw = os.path.join(args.query_dir, "Harbin_test.queries")
    if not os.path.exists(train_raw) or not os.path.exists(test_raw):
        raise FileNotFoundError("Query files not found. Run scripts/prepare_data.py first.")

    # ---- Data loading ----
    need_filtering = args.min_asym_ratio > 0 or args.min_dist > 0
    use_full = (args.train_size == 0) and not need_filtering

    if use_full:
        print("Loading full dataset (no filtering)...")
        train_data = read_query_file(train_raw, force_shift=args.force_shift)
        test_data = read_query_file(test_raw, force_shift=args.force_shift)
        train_rows, test_rows = train_data, test_data
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    else:
        print(f"Sampling {'asymmetric ' if need_filtering else ''}subset: "
              f"train={args.train_size}, test={args.test_size}")
        train_rows = select_subset(train_raw, args.train_size, args.min_asym_ratio,
                                   args.pool_multiplier, args.seed, min_dist=args.min_dist)
        test_rows = select_subset(test_raw, args.test_size, args.min_asym_ratio,
                                  args.pool_multiplier, args.seed + 1, min_dist=args.min_dist)
        describe_rows(train_rows, "Train")
        describe_rows(test_rows, "Test")

        train_file = filtered_dir / "Harbin_train.queries"
        test_file = filtered_dir / "Harbin_test.queries"
        write_query_file(str(train_file), train_rows)
        write_query_file(str(test_file), test_rows)

        train_data = read_query_file(str(train_file), force_shift=0)
        test_data = read_query_file(str(test_file), force_shift=0)

    max_distance = max(max(r[2] for r in train_data),
                       max(r[2] for r in test_data), 1.0)
    print(f"Max distance: {max_distance:.1f}")

    # ---- Graph loading ----
    graph = load_graph(args.data_dir, force_shift=args.force_shift)
    node_attributes = np.array([graph.nodes[n]["feature"] for n in graph.nodes()], dtype=np.float32)
    edge_attributes = np.array([[u, v, data["weight"]] for u, v, data in graph.edges(data=True)], dtype=np.float32)

    # ---- Dataloaders (no replicate for small data, each sample is unique) ----
    train_dataset = WorkloadDataset(train_data)
    test_dataset = WorkloadDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=get_num_workers(), pin_memory=device.startswith("cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=get_num_workers(), pin_memory=device.startswith("cuda"))

    # ---- Only 2 configs: L1+Split vs L1Tilde+Split ----
    configs = [
        {"metric": "l1",      "label": "L1      + Split"},
        {"metric": "l1tilde", "label": "L1Tilde + Split"},
    ]

    all_results = []

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  {cfg['label']}")
        print(f"{'=' * 60}")

        model = SAGEDistanceModel(
            n_input=node_attributes.shape[1],
            n_hidden_1=args.hidden_dim,
            n_hidden_2=args.embedding_dim,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            max_distance=max_distance,
            l1tilde_r=args.l1tilde_r,
            l1tilde_s=args.l1tilde_s,
            metric=cfg["metric"],
        )

        criterion = get_criterion(args.loss, model)
        optimizer = get_optimizer("adam", model, args.learning_rate)

        print(f"Training ({args.epochs} epochs, {len(train_dataset)} samples)...")
        history = model.fit(
            train_loader, criterion, optimizer,
            val_dataloader=test_loader if args.validate else None,
            epochs=args.epochs, display_step=10, device=device,
            fast_dev_run=args.fast_dev_run,
        )

        predictions, targets, query_latency = model.evaluate(
            test_loader, device=device, verbose=False)
        mre = float(np.mean(np.abs(predictions - targets) / np.maximum(targets, 1e-6)))
        print(f"  Test MRE: {mre:.4%}  |  Latency: {query_latency * 1e6:.1f} us/query")
        # Per-bucket breakdown
        bins = [0, 1000, 5000, 20000, 50000, 100000, 200000, 1e9]
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (targets >= lo) & (targets < hi)
            if mask.sum() > 5:
                mre_b = np.mean(np.abs(predictions[mask] - targets[mask]) / np.maximum(targets[mask], 1e-6))
                mae_b = np.mean(np.abs(predictions[mask] - targets[mask]))
                print(f"    [{lo:>7.0f}, {hi:>7.0f}): n={mask.sum():>5}, MRE={mre_b:.1%}, MAE={mae_b:.0f}m")

        all_results.append({
            "label": cfg["label"], "metric": cfg["metric"],
            "mre_percent": mre * 100.0,
            "query_latency_us": query_latency * 1_000_000,
            "val_mre_history": history["val_mre_epoch_history"],
        })

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  COMPARISON: L1 vs L1Tilde (Split only)")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Final MRE':>10} {'Best MRE':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")
    for r in all_results:
        best = min(r["val_mre_history"]) if r["val_mre_history"] else float("nan")
        print(f"{r['label']:<25} {r['mre_percent']:>9.2f}% {best:>9.2f}%")

    # Save results
    comparison = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": all_results,
        "train_subset": {"count": len(train_rows), "min_asym_ratio": args.min_asym_ratio},
        "test_subset": {"count": len(test_rows)},
    }
    with open(results_dir / "l1_vs_l1tilde_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to: {results_dir / 'l1_vs_l1tilde_comparison.json'}")


if __name__ == "__main__":
    main()

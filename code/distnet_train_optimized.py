"""
DistNet Dynamic Training Script (Memory Optimized)
--------------------------------------------------
此脚本专门针对无法生成或加载巨大 `preprocessed_indices.npy` 的情况。
它采用 "动态采样" (On-the-fly Sampling) 策略：
1. 不加载边列表，而是直接加载 SDM 矩阵 (支持 mmap，节省内存)。
2. 训练时随机采样 (u, v) 节点对，若 sdm[u,v] 有效则用于训练。
3. 验证集在开始时固定生成，以保证验证指标的可比性。

Usage:
    python distnet_train_dynamic.py --cities beijing --samples-per-epoch 1000000
"""

import argparse
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 尝试导入原项目依赖，确保兼容性
try:
    from distnet_model import ImprovedMultiLayerPerceptron
    from config import get_config
except ImportError:
    print("Error: 找不到 distnet_model.py 或 config.py。请确保此脚本在项目根目录下运行。")
    exit(1)

# -----------------------------------------------------------------------------
# 1. 动态数据集 (保持不变)
# -----------------------------------------------------------------------------
class DynamicDistanceDataset(Dataset):
    """
    不需要预先计算 indices 的数据集。
    在 __getitem__ 中随机采样 (u, v)。
    """
    def __init__(self, sdm, num_nodes, samples_per_epoch, max_dist=1.0):
        """
        :param sdm: (N, N) 距离矩阵 (numpy array or memmap)
        :param num_nodes: 节点数量
        :param samples_per_epoch: 每个 epoch 采样的样本数 (虚拟长度)
        :param max_dist: 用于归一化的最大距离
        """
        self.sdm = sdm
        self.num_nodes = num_nodes
        self.samples_per_epoch = samples_per_epoch
        self.max_dist = max_dist

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 随机重试直到找到有效对 (通常在 LSCC 中一次就能命中)
        # 为了防止死循环，设置最大尝试次数
        for _ in range(100):
            u = np.random.randint(0, self.num_nodes)
            v = np.random.randint(0, self.num_nodes)
            
            if u == v:
                continue
                
            dist = self.sdm[u, v]
            
            # 检查是否可达 (finite) 且非自身
            if np.isfinite(dist):
                # 返回 u, v, dist
                return torch.tensor(u, dtype=torch.long), \
                       torch.tensor(v, dtype=torch.long), \
                       torch.tensor(dist / self.max_dist, dtype=torch.float32)
        
        # 如果运气极差随机不到，返回一个默认值 (避免 crash)
        return torch.tensor(0, dtype=torch.long), \
               torch.tensor(1, dtype=torch.long), \
               torch.tensor(self.sdm[0, 1] / self.max_dist, dtype=torch.float32)

class StaticValidationDataset(Dataset):
    """
    验证集使用固定的索引，保证每个 Epoch 比较基准一致。
    """
    def __init__(self, indices, sdm, max_dist=1.0):
        self.indices = torch.from_numpy(indices).long()
        self.max_dist = max_dist
        # 预取标签并归一化
        self.targets = torch.from_numpy(sdm[indices[:, 0], indices[:, 1]] / max_dist).float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx, 0], self.indices[idx, 1], self.targets[idx]

# -----------------------------------------------------------------------------
# 2. 数据加载函数
# -----------------------------------------------------------------------------
def load_data_dynamic(config, city, data_root, samples_per_epoch=1000000, valid_size=50000):
    city_path = Path(data_root) / city / "pre"
    print(f"Loading data for {city} from {city_path}...")

    # A. 加载 Embedding
    embed_path = city_path / "preprocessed_embed.npy"
    if not embed_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    embed = np.load(embed_path).astype(np.float32)
    
    # B. 加载坐标 (Long/Lat)
    # [Check] 确保加载的是归一化后的坐标 (preprocessed_node_long_lat.npy)
    # 而不是原始坐标 (preprocessed_node_long_lat_origin.npy)
    coord_path = city_path / "preprocessed_node_long_lat.npy"
    if not coord_path.exists():
        raise FileNotFoundError(f"Coords file not found: {coord_path}")
    
    # coords 应该是经过 Z-score 归一化的: (x - mean) / std
    coords = np.load(coord_path).astype(np.float32)
    print(f"Coords loaded. Shape: {coords.shape} (Ensured Normalized)")

    # 合并特征: [Embedding, Coords]
    node_features_np = np.hstack([embed, coords])
    num_nodes = node_features_np.shape[0]
    feature_dim = node_features_np.shape[1]
    
    # 转为 Tensor
    node_features = torch.from_numpy(node_features_np)
    
    print(f"Nodes: {num_nodes}, Feature Dim: {feature_dim}")

    # C. 加载 SDM (使用 mmap_mode 以节省内存)
    sdm_path = city_path / "preprocessed_sdm.npy"
    if not sdm_path.exists():
        raise FileNotFoundError(f"SDM file not found: {sdm_path}")
    
    print("Loading SDM (Memory Mapped)...")
    sdm = np.load(sdm_path, mmap_mode='r')
    
    # 计算最大距离用于归一化 (防止 Loss 过大)
    print("Calculating max distance for normalization...")
    max_dist = np.max(sdm[np.isfinite(sdm)])
    print(f"Max distance: {max_dist}")

    # D. 构建数据集
    print(f"Creating Dynamic Training Dataset (Virtual Size: {samples_per_epoch})...")
    batch_size = config.get('batch_size', 1024)
    
    train_dataset = DynamicDistanceDataset(sdm, num_nodes, samples_per_epoch, max_dist=max_dist)
    
    # E. 构建固定验证集
    print(f"Generating Fixed Validation Set ({valid_size} samples)...")
    valid_indices_list = []
    count = 0
    while count < valid_size:
        us = np.random.randint(0, num_nodes, size=valid_size * 2)
        vs = np.random.randint(0, num_nodes, size=valid_size * 2)
        
        for u, v in zip(us, vs):
            if u == v: continue
            dist = sdm[u, v]
            if np.isfinite(dist):
                valid_indices_list.append([u, v])
                count += 1
                if count >= valid_size:
                    break
    
    valid_indices = np.array(valid_indices_list, dtype=np.int32)
    valid_dataset = StaticValidationDataset(valid_indices, sdm, max_dist=max_dist)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, valid_loader, node_features, feature_dim

# -----------------------------------------------------------------------------
# 3. 训练函数 (已更新 Device 逻辑)
# -----------------------------------------------------------------------------
def train_model(train_loader, valid_loader, node_features, config, city, out_dir, save_prefix):
    # [Device Selection]: Cuda -> MPS (Mac) -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA (NVIDIA)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # 准备模型参数
    feature_dim = node_features.shape[1]
    hidden_dim = config.get('hidden_dim', 512)
    n_output = config.get('n_output', 64) 
    
    print(f"Initializing Model: Input={feature_dim*2}, Hidden={hidden_dim}, Output={n_output}")
    
    # 参数顺序: (n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    model = ImprovedMultiLayerPerceptron(
        feature_dim * 2, 
        hidden_dim, 
        hidden_dim, 
        hidden_dim, 
        n_output
    ).to(device)
    
    lr = config.get('learning_rate', config.get('lr', 0.001))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 1e-5))
    criterion = nn.MSELoss()
    
    # 将特征移入显存
    node_features = node_features.to(device)

    best_val_loss = float('inf')
    early_stop_count = 0
    patience = config.get('patience', 10)
    num_epochs = config.get('num_epoch', config.get('epochs', 50))
    
    # 根据 config.type 构造文件名后缀，匹配项目命名规范
    model_type = config.get('type', 2)
    if model_type == 1:
        type_suffix = "_1"
    elif model_type == 2:
        type_suffix = "_tilde_L1"
    elif model_type == 3:
        type_suffix = "_L1"
    else:
        type_suffix = ""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(out_dir) / f"{save_prefix}_{city}{type_suffix}.ckpt"

    print("Start Training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (u, v, target) in enumerate(train_loader):
            u, v, target = u.to(device), v.to(device), target.to(device)
            
            # 提取特征
            feat_u = node_features[u]
            feat_v = node_features[v]
            
            optimizer.zero_grad()
            
            # 传递两个独立的特征张量
            output = model(feat_u, feat_v).squeeze()
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')

        avg_train_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for u, v, target in valid_loader:
                u, v, target = u.to(device), v.to(device), target.to(device)
                feat_u = node_features[u]
                feat_v = node_features[v]
                
                output = model(feat_u, feat_v).squeeze()
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(valid_loader)
        
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished in {time.time() - start_time:.2f}s")

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=["beijing"], help="City names to train on")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./param")
    parser.add_argument("--save-prefix", default="distnet_best")
    parser.add_argument("--samples-per-epoch", type=int, default=1000000, help="Number of random pairs to sample per epoch")
    args, _ = parser.parse_known_args()

    # 获取默认配置并处理 Namespace 兼容性
    try:
        config, _ = get_config()
        if isinstance(config, argparse.Namespace):
            config = vars(config)
    except:
        config = {
            'learning_rate': 0.001,
            'num_epoch': 50,
            'batch_size': 1024,
            'hidden_dim': 512,
            'patience': 10,
            'n_output': 64,
            'type': 2
        }
        print("Warning: Using fallback config dictionary.")

    for city in args.cities:
        print(f"\n===== Starting Dynamic Training for {city} =====")
        try:
            train_loader, valid_loader, node_features, _ = load_data_dynamic(
                config, city, args.data_root, samples_per_epoch=args.samples_per_epoch
            )
            train_model(train_loader, valid_loader, node_features, config, city, args.out_dir, args.save_prefix)
        except Exception as e:
            print(f"Failed to train on {city}: {e}")
            import traceback
            traceback.print_exc()
"""
本文件为Distnet模型的最完整代码，包含数据预处理、模型训练、微调以及验证等完整流程。
Distnet用于路网距离估计，结合节点嵌入和地理坐标信息，通过神经网络进行距离预测。
用到的技术包括：
1. tilde-L1正则化以提升模型针对有向图的适应性；
2. landmark地标节点增强训练以提升模型性能；
3. fine-tune微调高误差样本以提升整体精度；
4. dist2vec+经纬度节点嵌入以提供丰富的节点特征信息；
5. 每个epoch随机采样训练数据以加快训练速度。
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split

from distnet_model import ImprovedMultiLayerPerceptron
from distnet_model import farthest_selection

from config import get_config


class DistanceDataset(Dataset):
    def __init__(self, indices, embed, node_long_lat, sdm, LM_indices=None):
        self.indices = indices
        self.embed = embed
        self.node_long_lat = node_long_lat
        self.sdm = sdm
        self.LM_indices = LM_indices if LM_indices else []

    def __len__(self):
        return len(self.indices) + len(self.LM_indices)

    def __getitem__(self, idx):
        if idx < len(self.indices):
            i, j = self.indices[idx]
        else:
            i, j = self.LM_indices[idx - len(self.indices)]
        x1 = np.concatenate((self.embed[i], self.node_long_lat[i]), axis=0)
        x2 = np.concatenate((self.embed[j], self.node_long_lat[j]), axis=0)
        y = self.sdm[i][j]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(y,
                                                                                                          dtype=torch.float32)


def load_and_preprocess_data(config, city: str = "harbin", data_root: str = "/home/lizhuoran/distance/data"):
    """
    Load and preprocess data for training
    :param config: Configuration parameters
    :return: train_loader, valid_loader
    """
    # 直接加载预处理数据，优先从 city 专用目录 data/<city>/pre/
    from pathlib import Path
    pre_dir = Path(data_root) / city / "pre"
    # backward compatible fallback
    if not pre_dir.exists():
        pre_dir = Path(data_root) / "pre"

    sdm = np.load(pre_dir / "preprocessed_sdm.npy")
    embed = np.load(pre_dir / "preprocessed_embed.npy")
    node_long_lat = np.load(pre_dir / "preprocessed_node_long_lat.npy")
    node_long_lat_origin = np.load(pre_dir / "preprocessed_node_long_lat_origin.npy")  # 新增加载
    indices = np.load(pre_dir / "preprocessed_indices.npy", allow_pickle=True)
    LM_indices = np.load(pre_dir / "preprocessed_LM_indices.npy", allow_pickle=True)

    # Print city information
    print(f"{city} with {sdm.shape[1]} nodes")

    # 转换为list，便于后续采样和索引
    indices = indices.tolist()
    LM_indices = LM_indices.tolist()

    # Set random seed for reproducibility
    np.random.seed(42)
    perm = np.random.permutation(len(indices))
    train_size = int(len(indices) * 0.9)
    train_indices = [indices[i] for i in perm[:train_size]]
    valid_indices = [indices[i] for i in perm[train_size:]]

    # 删除以下重复地标采样和地标对生成代码
    # num_landmarks = max(int(sdm.shape[0] * 0.01), 20)
    # landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)
    # LM_indices = []
    # for i in range(len(landmark_indices)):
    #     for j in range(len(landmark_indices)):
    #         if sdm[landmark_indices[i]][landmark_indices[j]] != 0:
    #             LM_indices.append((landmark_indices[i], landmark_indices[j]))

    # Create datasets
    train_dataset = DistanceDataset(train_indices, embed, node_long_lat, sdm, LM_indices)
    valid_dataset = DistanceDataset(valid_indices, embed, node_long_lat, sdm, LM_indices=[])  # 验证集不加地标对

    # Data loaders（不再采样，全部用）
    batch_size = config.batch_size
    import os
    # num_workers用8个
    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, config, city: str = "chengdu", out_dir: str = "param", save_prefix: str = "distnet_best"):
    """
    Train the neural network model
    :param train_loader: Training data loader
    :param valid_loader: Validation data loader
    :param config: Configuration parameters
    """
    # 优先使用CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Model initialization
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim

    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Training parameters
    num_epochs = config.num_epoch
    learning_rate = config.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # if config.type == 2:
    #     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Get dataset from loader
    train_dataset = train_loader.dataset
    batch_size = train_loader.batch_size

    # Training tracking variables
    min_loss = 100
    start_time = time.time()
    early_stop = 0

    # create a targeted random num generator
    rng = np.random.RandomState(42)  # fixed seed

    # Training loop
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        total_train_samples = 0
        for batch_x1, batch_x2, batch_y in train_loader:
            batch_x1 = batch_x1.to(device, non_blocking=True)
            batch_x2 = batch_x2.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x1, batch_x2)
                    loss = criterion(outputs, batch_y.unsqueeze(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * batch_x1.size(0)  # 优化累加方式
            total_train_samples += batch_x1.size(0)
        train_loss /= total_train_samples

        # Identify high-error samples
        # model.eval()
        # loss_list = []
        # with torch.no_grad():
        #     for batch_x1, batch_x2, batch_y in train_loader:
        #         batch_x1 = batch_x1.to(device, non_blocking=True)
        #         batch_x2 = batch_x2.to(device, non_blocking=True)
        #         batch_y = batch_y.to(device, non_blocking=True)
        #         if scaler:
        #             with torch.amp.autocast('cuda'):
        #                 outputs = model(batch_x1, batch_x2)
        #                 loss = criterion(outputs, batch_y.unsqueeze(-1))
        #         else:
        #             outputs = model(batch_x1, batch_x2)
        #             loss = criterion(outputs, batch_y.unsqueeze(-1))
        #         loss_list.extend([(loss.item(), (x1, x2, y)) for x1, x2, y in zip(batch_x1, batch_x2, batch_y)])

        # Fine-tuning phase
        # loss_list.sort(reverse=True, key=lambda x: x[0])
        # high_error_samples = [sample for _, sample in loss_list[:int(0.1 * len(loss_list))]]

        # model.train()
        # fine_tune_loss = 0
        # fine_tune_batch_size = 32
        # fine_tune_num_batches = 0

        # for i in range(0, len(high_error_samples), fine_tune_batch_size):
        #     batch = high_error_samples[i:i + fine_tune_batch_size]
        #     x1 = torch.stack([x[0] for x in batch]).to(device, non_blocking=True)
        #     x2 = torch.stack([x[1] for x in batch]).to(device, non_blocking=True)
        #     y = torch.stack([x[2] for x in batch]).to(device, non_blocking=True)
        #     optimizer.zero_grad(set_to_none=True)
        #     if scaler:
        #         with torch.amp.autocast('cuda'):
        #             outputs = model(x1, x2)
        #             loss = criterion(outputs, y.unsqueeze(-1))
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         outputs = model(x1, x2)
        #         loss = criterion(outputs, y.unsqueeze(-1))
        #         loss.backward()
        #         optimizer.step()
        #     fine_tune_loss += loss.item()
        #     fine_tune_num_batches += 1

        # fine_tune_loss /= fine_tune_num_batches

        # Validation phase
        model.eval()
        valid_loss = 0
        total_valid_samples = 0
        with torch.no_grad():
            for batch_x1, batch_x2, batch_y in valid_loader:
                batch_x1 = batch_x1.to(device, non_blocking=True)
                batch_x2 = batch_x2.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_x1, batch_x2)
                        loss = criterion(outputs, batch_y.unsqueeze(-1))
                else:
                    outputs = model(batch_x1, batch_x2)
                    loss = criterion(outputs, batch_y.unsqueeze(-1))
                valid_loss += loss.item() * batch_x1.size(0)  # 优化累加方式
                total_valid_samples += batch_x1.size(0)
        valid_loss /= total_valid_samples

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Print progress
        print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Valid Loss: {valid_loss:.8f}, Time: {epoch_duration:.2f}s')

        # Model checkpointing and early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss
            import os
            os.makedirs(out_dir, exist_ok=True)
            # build filename with city and config.type
            if config.type == 1:
                fname = os.path.join(out_dir, f"{save_prefix}_{city}_1.ckpt")
            elif config.type == 2:
                fname = os.path.join(out_dir, f"{save_prefix}_{city}_tilde_L1.ckpt")
            elif config.type == 3:
                fname = os.path.join(out_dir, f"{save_prefix}_{city}_L1.ckpt")
            else:
                fname = os.path.join(out_dir, f"{save_prefix}_{city}.ckpt")
            torch.save(model.state_dict(), fname)
            print(f'Model saved to {fname}')
            # also append best validation info to per-city log
            log_dir = os.path.join("log")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{save_prefix}_{city}_train.log")
            with open(log_file, 'a') as lf:
                lf.write(f'Epoch: {epoch+1}, Train Loss: {train_loss:.8f}, Valid Loss: {valid_loss:.8f}, Time: {epoch_duration:.2f}s\n')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > 5:
                break

    print("Optimization Finished!")
    end_time = time.time()
    print("Training time: ", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistNet on one or more city's preprocessed data")
    parser.add_argument("--cities", nargs="+", default=["chengdu"], help="One or more city names to train on (space-separated), e.g. --cities harbin porto")
    parser.add_argument("--data-root", type=str, default="/home/lizhuoran/distance/data", help="Root data directory containing city folders")
    parser.add_argument("--out-dir", type=str, default="/home/lizhuoran/distance/param", help="Directory to save model checkpoints")
    parser.add_argument("--save-prefix", type=str, default="distnet_best", help="Prefix for saved checkpoint filenames")
    args = parser.parse_args()

    config, _ = get_config()
    for city in args.cities:
        print(f"\n===== Starting training for city: {city} =====")
        train_loader, valid_loader = load_and_preprocess_data(config, city=city, data_root=args.data_root)
        train_model(train_loader, valid_loader, config, city=city, out_dir=args.out_dir, save_prefix=args.save_prefix)
        # free CUDA memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('\nAll trainings completed.')
import numpy as np
import pandas as pd
import pickle
import time
# from tqdm import tqdm   # 移除tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distnet_model import ImprovedMultiLayerPerceptron
from distnet_model import farthest_selection

from config import get_config


class DistanceDataset(Dataset):
    def __init__(self, indices, embed, node_long_lat, sdm):
        self.indices = indices
        self.embed = embed
        self.node_long_lat = node_long_lat
        self.sdm = sdm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        x1 = np.concatenate((self.embed[i], self.node_long_lat[i]), axis=0)
        x2 = np.concatenate((self.embed[j], self.node_long_lat[j]), axis=0)
        y = self.sdm[i][j]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_and_preprocess_data(config):
    """
    Load and preprocess data for training
    :param config: Configuration parameters
    :return: train_loader, valid_loader
    """
    # Load shortest distance matrix
    file_name = "data/chengdu_directed_shortest_distance_matrix.npy"
    sdm = np.load(file_name)

    # Print city information
    print("chengdu with " + str(sdm.shape[1]) + " nodes")

    # Load dist2vec embeddings
    with open("param/dist2vec_embed.pkl", 'rb') as f:
        embed = pickle.load(f)

    # Load node coordinates
    node_long_lat = pd.read_csv("data/chengdu_node-mod.txt", header=0, sep=',')
    node_long_lat_origin = np.array(node_long_lat)[:, 1:3]
    node_long_lat = np.array(node_long_lat)[:, 1:3]

    # Normalize data
    maxLength = np.max(sdm)
    sdm = sdm / maxLength

    embed = np.array(list(embed.values()))
    embed = (embed - embed.min()) / (embed.max() - embed.min())

    node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (
                node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
    node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (
                node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

    # Create index pairs for non-zero distances
    indices = []
    for i in range(sdm.shape[0]):
        for j in range(sdm.shape[1]):
            if sdm[i][j] != 0.0:
                indices.append((i, j))

    # Set random seed for reproducibility
    np.random.seed(42)
    indices = np.array(indices)
    perm = np.random.permutation(len(indices))
    train_size = int(len(indices) * 0.9)
    train_indices = indices[perm[:train_size]]
    valid_indices = indices[perm[train_size:]]

    # 只用原始数据集，不用LM_indices
    train_dataset = DistanceDataset(train_indices, embed, node_long_lat, sdm)
    valid_dataset = DistanceDataset(valid_indices, embed, node_long_lat, sdm)

    # Data loaders（优化参数）
    batch_size = config.batch_size
    # 用8和物理核心数中的较小值作为num_workers
    num_workers = min(8, os.cpu_count() or 1)  # 使用全部CPU核心
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # 优化
        num_workers=num_workers,
        persistent_workers=True  # PyTorch 2.8.0支持
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,  # 优化
        num_workers=num_workers,
        persistent_workers=True
    )

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, config):
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
        torch.backends.cudnn.benchmark = True  # 加速

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

    # Mixed Precision Training Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None  # AMP only for CUDA

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
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
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0
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
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Print progress
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Valid Loss: {valid_loss:.8f}, Time: {epoch_duration:.2f}s')

        # Model checkpointing and early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss
            if config.type == 1:
                torch.save(model.state_dict(), "param/distnet_base_best_chengdu_1.ckpt")
            elif config.type == 2:
                torch.save(model.state_dict(), "param/distnet_base_best_chengdu_tilde_L1.ckpt")
            elif config.type == 3:
                torch.save(model.state_dict(), "param/distnet_base_best_chengdu_L1.ckpt")
            print('Model saved.')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > 5:
                break

    print("Optimization Finished!")
    end_time = time.time()
    print("Training time: ", end_time - start_time)


if __name__ == "__main__":
    config, _ = get_config()
    # 打印type
    print(f"Model type: {config.type}")
    train_loader, valid_loader = load_and_preprocess_data(config)
    train_model(train_loader, valid_loader, config)
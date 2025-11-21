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

import numpy as np
import pandas as pd
import pickle
import time
from tqdm import tqdm

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

    # Select landmarks
    num_landmarks = max(int(sdm.shape[0] * 0.01), 20)
    landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)

    # Create landmark pairs
    LM_indices = []
    for i in range(len(landmark_indices)):
        for j in range(len(landmark_indices)):
            if sdm[landmark_indices[i]][landmark_indices[j]] != 0:
                LM_indices.append((landmark_indices[i], landmark_indices[j]))

    # Create datasets
    train_dataset = DistanceDataset(train_indices, embed, node_long_lat, sdm, LM_indices)
    valid_dataset = DistanceDataset(valid_indices, embed, node_long_lat, sdm)

    # Data loaders（不再采样，全部用）
    batch_size = config.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, config):
    """
    Train the neural network model
    :param train_loader: Training data loader
    :param valid_loader: Validation data loader
    :param config: Configuration parameters
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # print(f'Using device: {device}')

    device = torch.device('cpu')

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
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x1, batch_x2, batch_y in tqdm(train_loader):
            # Forward pass
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Identify high-error samples
        model.eval()
        loss_list = []
        with torch.no_grad():
            for batch_x1, batch_x2, batch_y in train_loader:
                batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))
                loss_list.extend([(loss.item(), (x1, x2, y)) for x1, x2, y in zip(batch_x1, batch_x2, batch_y)])

        # Fine-tuning phase
        loss_list.sort(reverse=True, key=lambda x: x[0])
        high_error_samples = [sample for _, sample in loss_list[:int(0.1 * len(loss_list))]]

        model.train()
        fine_tune_loss = 0
        fine_tune_batch_size = 32
        fine_tune_num_batches = 0

        # Fine-tune on high-error samples
        for i in range(0, len(high_error_samples), fine_tune_batch_size):
            batch = high_error_samples[i:i + fine_tune_batch_size]
            x1 = torch.stack([x[0] for x in batch])
            x2 = torch.stack([x[1] for x in batch])
            y = torch.stack([x[2] for x in batch])
            outputs = model(x1, x2)
            loss = criterion(outputs, y.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fine_tune_loss += loss.item()
            fine_tune_num_batches += 1

        fine_tune_loss /= fine_tune_num_batches

        # Validation phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_x1, batch_x2, batch_y in valid_loader:
                batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        # Print progress
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Fine-tune Loss: {fine_tune_loss:.8f}, Valid Loss: {valid_loss:.8f}')

        # Model checkpointing and early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss
            if config.type == 1:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_1.ckpt")
            elif config.type == 2:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_tilde_L1.ckpt")
            elif config.type == 3:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_L1.ckpt")
            print('Model saved.')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > 5:
                break

    print("Optimization Finished!")
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    # 评估模型并保存结果
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in valid_loader:
            batch_x1, batch_x2 = batch_x1.to(device), batch_x2.to(device)
            outputs = model(batch_x1, batch_x2).cpu().numpy().flatten()
            y_pred.extend(outputs)
            y_true.extend(batch_y.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    abs_errors = np.abs(y_true - y_pred)
    global_abs = np.mean(abs_errors)
    max_abs = np.max(abs_errors)
    min_abs = np.min(abs_errors)
    rel_errors = abs_errors / (y_true + 1e-8)
    global_rel = np.mean(rel_errors)
    max_rel = np.max(rel_errors)
    min_rel = np.min(rel_errors)

    # 保存评估结果
    with open("log/distnet.txt", "w") as f:
        f.write(f"Mean Square Error: {mse:.5f}\n")
        f.write(f"Mean Absolute Error: {global_abs:.5f}\n")
        f.write(f"Max Absolute Error: {max_abs:.5f}\n")
        f.write(f"Min Absolute Error: {min_abs:.5f}\n")
        f.write(f"Mean Relative Error: {global_rel:.8f}\n")
        f.write(f"Max Relative Error: {max_rel:.5f}\n")
        f.write(f"Min Relative Error: {min_rel:.5f}\n")

    print("\nDistnet training and evaluation completed!")


if __name__ == "__main__":
    config, _ = get_config()
    train_loader, valid_loader = load_and_preprocess_data(config)
    train_model(train_loader, valid_loader, config)
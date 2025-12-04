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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split

from distnet_model import ImprovedMultiLayerPerceptron, LearnableDistNet
from distnet_model import farthest_selection

from config import get_config


class DistanceDataset(Dataset):
    def __init__(self, indices, embed, node_long_lat, sdm, LM_indices=None):
        self.indices = indices
        self.embed = embed # Keep for compatibility, but we will use indices
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
        
        # Return indices and coordinates separately
        # x1 = np.concatenate((self.embed[i], self.node_long_lat[i]), axis=0)
        # x2 = np.concatenate((self.embed[j], self.node_long_lat[j]), axis=0)
        
        coords1 = self.node_long_lat[i]
        coords2 = self.node_long_lat[j]
        y = self.sdm[i][j]
        
        return torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long), \
               torch.tensor(coords1, dtype=torch.float32), torch.tensor(coords2, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)


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
    import os
    num_workers = min(8, os.cpu_count() or 1)
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

    return train_loader, valid_loader, embed


def train_model(train_loader, valid_loader, embed, config):
    """
    Train the neural network model
    :param train_loader: Training data loader
    :param valid_loader: Validation data loader
    :param embed: Pretrained embeddings for initialization
    :param config: Configuration parameters
    """
    # 优先使用CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Model initialization
    # embed_dim = config.embed_dim
    # long_lat_embed_dim = config.long_lat_embed_dim
    # input_dim = embed_dim + long_lat_embed_dim

    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    
    # Use LearnableDistNet
    n_nodes = embed.shape[0]
    model = LearnableDistNet(n_nodes, embed, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)
    # model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Training parameters
    num_epochs = 100 # Increase epochs
    learning_rate = config.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # if config.type == 2:
    #     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Use Log-MSE Loss to balance MRE and MAE optimization
    # Optimizing MSE in log-space is approximately equivalent to optimizing Relative Error,
    # but much more numerically stable than direct division.
    class LogMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()

        def forward(self, pred, target):
            return self.mse(torch.log1p(pred), torch.log1p(target))

    criterion = LogMSELoss()

    # Get dataset from loader
    train_dataset = train_loader.dataset
    batch_size = train_loader.batch_size

    # Training tracking variables
    min_mre = 100.0 # Track min MRE
    start_time = time.time()
    early_stop_counter = 0
    patience = 10 # Increase patience

    # History for plotting
    history = {
        'train_loss': [],
        'valid_loss': [],
        'mre': [],
        'mae': []
    }

    # create a targeted random num generator
    rng = np.random.RandomState(42)  # fixed seed

    # Training loop
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for batch_idx1, batch_idx2, batch_coords1, batch_coords2, batch_y in train_loader:
            batch_idx1 = batch_idx1.to(device, non_blocking=True)
            batch_idx2 = batch_idx2.to(device, non_blocking=True)
            batch_coords1 = batch_coords1.to(device, non_blocking=True)
            batch_coords2 = batch_coords2.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            target = batch_y.unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_idx1, batch_idx2, batch_coords1, batch_coords2)
                    loss = criterion(outputs, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_idx1, batch_idx2, batch_coords1, batch_coords2)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0
        total_mre = 0.0
        total_mae = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx1, batch_idx2, batch_coords1, batch_coords2, batch_y in valid_loader:
                batch_idx1 = batch_idx1.to(device, non_blocking=True)
                batch_idx2 = batch_idx2.to(device, non_blocking=True)
                batch_coords1 = batch_coords1.to(device, non_blocking=True)
                batch_coords2 = batch_coords2.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                target = batch_y.unsqueeze(-1)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_idx1, batch_idx2, batch_coords1, batch_coords2)
                        loss = criterion(outputs, target)
                else:
                    outputs = model(batch_idx1, batch_idx2, batch_coords1, batch_coords2)
                    loss = criterion(outputs, target)
                
                valid_loss += loss.item()
                
                # Calculate metrics
                pred = torch.abs(outputs) 
                
                # MAE
                mae = torch.abs(pred - target)
                total_mae += torch.sum(mae).item()
                
                # MRE
                # Add epsilon to denominator to avoid division by zero
                mre = torch.abs(pred - target) / (target + 1e-7)
                total_mre += torch.sum(mre).item()
                
                total_samples += batch_y.size(0)

        valid_loss /= len(valid_loader)
        avg_mre = total_mre / total_samples
        avg_mae = total_mae / total_samples

        # Update history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['mre'].append(avg_mre)
        history['mae'].append(avg_mae)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Print progress
        log_msg = f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Valid Loss: {valid_loss:.8f}, MRE: {avg_mre:.8f}, MAE: {avg_mae:.8f}, Time: {epoch_duration:.2f}s'
        print(log_msg)
        
        # Write to log file
        with open("log/distnet_training_log.txt", "a") as f:
            f.write(log_msg + "\n")

        # Model checkpointing and early stopping
        # Use MRE for model selection
        if avg_mre < min_mre:
            min_mre = avg_mre
            if config.type == 1:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_1.ckpt")
            elif config.type == 2:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_tilde_L1.ckpt")
            elif config.type == 3:
                torch.save(model.state_dict(), "param/distnet_best_chengdu_L1.ckpt")
            print('Model saved.')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > patience:
                print("Early stopping triggered.")
                break

    print("Optimization Finished!")
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Valid Loss')
        plt.title('Loss History')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(history['mre'], label='MRE', color='orange')
        plt.title('Mean Relative Error')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(history['mae'], label='MAE', color='green')
        plt.title('Mean Absolute Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('figure/training_history.png')
        print("Training history plot saved to figure/training_history.png")
    except Exception as e:
        print(f"Failed to plot history: {e}")


if __name__ == "__main__":
    config, _ = get_config()
    train_loader, valid_loader, embed = load_and_preprocess_data(config)
    train_model(train_loader, valid_loader, embed, config)
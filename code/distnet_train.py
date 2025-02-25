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
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_preprocess_data(config):
    """
    load and preprocess data
    :param sdm: 最短距离矩阵
    :param embed: 嵌入向量
    :param node_long_lat: 节点经纬度
    :param input_dim: 输入维度
    :param indices: 数据索引
    :param LM_indices: 地标索引
    :return: train_loader, valid_loader, test_loader
    """

    # load data
    file_name = "data/chengdu_directed_shortest_distance_matrix.npy"
    sdm = np.load(file_name)

    #  print cite name
    print("chengdu with " + str(sdm.shape[1]) + " nodes")

    # use pickle to load node2vec_embed
    with open("param/dist2vec_embed.pkl", 'rb') as f:
        embed = pickle.load(f)

    # load node long lat
    node_long_lat = pd.read_csv("data/chengdu_node-mod.txt", header=0, sep=',')
    node_long_lat_origin = np.array(node_long_lat)[:, 1:3]
    node_long_lat = np.array(node_long_lat)[:, 1:3]

    # Normalization
    maxLength = np.max(sdm)
    sdm = sdm / maxLength

    embed = np.array(list(embed.values()))
    embed = (embed - embed.min()) / (embed.max() - embed.min())

    node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
    node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

    # consturct the train, valid, test dataset
    indices = []
    for i in range(sdm.shape[0]):
        for j in range(sdm.shape[1]):
            if sdm[i][j] != 0.0:
                indices.append((i, j))

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the indices
    np.random.shuffle(indices)

    # set the landmark numbers
    num_landmarks = max(int(sdm.shape[0] * 0.01), 20)
    # obtain the landmark indices
    landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)
    LM_indices = []
    for i in range(len(landmark_indices)):
        for j in range(len(landmark_indices)):
            if sdm[landmark_indices[i]][landmark_indices[j]] != 0:  # only select the different landmarks
                LM_indices.append((landmark_indices[i], landmark_indices[j]))

    # split the indices
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # construct the DistanceDataset
    train_dataset = DistanceDataset(train_indices, embed, node_long_lat, sdm, LM_indices)
    valid_dataset = DistanceDataset(valid_indices, embed, node_long_lat, sdm)
    test_dataset = DistanceDataset(test_indices, embed, node_long_lat, sdm)

    # construct the DataLoader
    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_model(train_loader, valid_loader, config):
    """
    训练模型
    :return: None
    """
    device = torch.device('cpu')

    # Initialize your model
    # parameter setting
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim

    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = 1
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Define training parameters
    num_epochs = config.num_epoch
    learning_rate = config.learning_rate

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    min_loss = 100
    start_time = time.time()

    # train process
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x1, batch_x2, batch_y in train_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
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

        # Sort by loss and select high-error samples
        loss_list.sort(reverse=True, key=lambda x: x[0])
        high_error_samples = [sample for _, sample in
                              loss_list[:int(0.1 * len(loss_list))]]  # Top 10% high-error samples

        # Fine-tune on high-error samples
        model.train()
        fine_tune_loss = 0
        for x1, x2, y in high_error_samples:
            x1, x2, y = x1.unsqueeze(0), x2.unsqueeze(0), y.unsqueeze(0)  # Add batch dimension
            outputs = model(x1, x2)
            loss = criterion(outputs, y.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fine_tune_loss += loss.item()

        fine_tune_loss /= len(high_error_samples)

        # valid process
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_x1, batch_x2, batch_y in valid_loader:
                batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y.unsqueeze(-1))
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        # print train_loss, fine_tune_loss & valid_loss
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Fine-tune Loss: {fine_tune_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        # If the current valid_loss is less than the previous best valid_loss, save the model.
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), "param/distnet_best_chengdu_tilde_L1.ckpt")
            print('Model saved.')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > 10:
                break

    print("Optimization Finished!")
    end_time = time.time()
    print("Training time: ", end_time - start_time)

if __name__ == "__main__":
    config, _ = get_config()
    train_loader, valid_loader, _ = load_and_preprocess_data(config)
    train_model(train_loader, valid_loader, config)
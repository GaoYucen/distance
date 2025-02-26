import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

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
    Load and preprocess data for testing
    :param config: Configuration parameters
    :return: test_loader
    """
    # Load shortest distance matrix
    file_name = "../data/chengdu_directed_shortest_distance_matrix.npy"
    sdm = np.load(file_name)

    # Print city information
    print("chengdu with " + str(sdm.shape[1]) + " nodes")

    # Load node2vec embeddings
    with open("../param/dist2vec_embed.pkl", 'rb') as f:
        embed = pickle.load(f)

    # Load node coordinates
    node_long_lat = pd.read_csv("../data/chengdu_node-mod.txt", header=0, sep=',')
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

    # Select landmarks
    num_landmarks = max(int(sdm.shape[0] * 0.01), 20)
    landmark_indices = farthest_selection(node_long_lat_origin, num_landmarks)

    # Create landmark pairs
    LM_indices = []
    for i in range(len(landmark_indices)):
        for j in range(len(landmark_indices)):
            if sdm[landmark_indices[i]][landmark_indices[j]] != 0:
                LM_indices.append((landmark_indices[i], landmark_indices[j]))

    # Create dataset
    test_dataset = DistanceDataset(indices, embed, node_long_lat, sdm, LM_indices)

    # Create test loader
    batch_size = config.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def test_model(test_loader, config):
    """
    Test the pre-trained model
    :param test_loader: Test data loader
    :param config: Configuration parameters
    """
    device = torch.device('cpu')

    # Model initialization
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim

    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = get_config()
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Load pre-trained model
    model.load_state_dict(torch.load("../param/distnet_best_chengdu_tilde_L1.ckpt", map_location=device))
    model.eval()  # Set model to evaluation mode

    criterion = torch.nn.MSELoss()

    # Evaluate on test data
    test_loss = 0
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in test_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    config, _ = get_config()
    test_loader = load_and_preprocess_data(config)
    test_model(test_loader, config)

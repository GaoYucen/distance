import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from distnet_model import ImprovedMultiLayerPerceptron, LearnableDistNet
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
        
        coords1 = self.node_long_lat[i]
        coords2 = self.node_long_lat[j]
        y = self.sdm[i][j]
        
        return torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long), \
               torch.tensor(coords1, dtype=torch.float32), torch.tensor(coords2, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)


def load_and_preprocess_data(config, eval_sample_size=10000):
    """
    Load and preprocess data for sampled evaluation
    :param config: Configuration parameters
    :param eval_sample_size: Number of samples for evaluation
    :return: sampled_eval_loader, maxLength
    """
    # Load shortest distance matrix
    file_name = "data/chengdu_directed_shortest_distance_matrix.npy"
    sdm = np.load(file_name)

    # Print city information
    print("chengdu with " + str(sdm.shape[1]) + " nodes")

    # Load node2vec embeddings
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
    indices = np.array(indices)

    # 随机采样全集中的若干合法节点对
    np.random.seed(123)
    sample_size = min(eval_sample_size, len(indices))
    sample_idxs = np.random.choice(len(indices), sample_size, replace=False)
    sampled_eval_indices = indices[sample_idxs]

    # Create dataset
    eval_dataset = DistanceDataset(sampled_eval_indices, embed, node_long_lat, sdm)

    # Create loader
    batch_size = config.batch_size
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return eval_loader, maxLength, embed


def test_model(eval_loader, config, maxLength, embed):
    """
    Test the pre-trained model and compute required metrics
    :param eval_loader: Test data loader
    :param config: Configuration parameters
    :param maxLength: scalar used to denormalize distances
    :param embed: Pretrained embeddings for initialization
    """
    # 优先使用CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model initialization
    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    
    n_nodes = embed.shape[0]
    model = LearnableDistNet(n_nodes, embed, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Load pre-trained model
    if config.type == 1:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_1.ckpt", map_location=device))
    elif config.type == 2:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_tilde_L1.ckpt", map_location=device))
    elif config.type == 3:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_L1.ckpt", map_location=device))

    model.eval()  # Set model to evaluation mode

    # Collect predictions and truths (denormalized)
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_idx1, batch_idx2, batch_coords1, batch_coords2, batch_y in eval_loader:
            batch_idx1 = batch_idx1.to(device)
            batch_idx2 = batch_idx2.to(device)
            batch_coords1 = batch_coords1.to(device)
            batch_coords2 = batch_coords2.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_idx1, batch_idx2, batch_coords1, batch_coords2)
            outputs_np = outputs.cpu().numpy().squeeze()
            trues_np = batch_y.cpu().numpy()
            # denormalize
            outputs_np = outputs_np * maxLength
            trues_np = trues_np * maxLength
            # ensure shapes
            if outputs_np.ndim == 0:
                outputs_np = np.array([outputs_np])
            if trues_np.ndim == 0:
                trues_np = np.array([trues_np])

            all_preds.append(outputs_np.reshape(-1))
            all_trues.append(trues_np.reshape(-1))

    if len(all_preds) == 0:
        print("No test predictions.")
        return

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    abs_err = np.abs(preds - trues)
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(abs_err)
    max_abs = np.max(abs_err)
    min_abs = np.min(abs_err)

    # relative errors: define as 0 where true == 0 to avoid inf
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(trues > 0, abs_err / trues, 0.0)

    mean_rel = np.mean(rel)
    max_rel = np.max(rel)
    min_rel = np.min(rel)

    print(f"Eval sample size: {len(trues)}")
    print(f"mean square error: {mse:.2f}")
    print(f"mean absolute error: {mae:.5f}")
    print(f"max absolute error: {max_abs:.4f}")
    print(f"min absolute error: {min_abs:.1f}")
    print(f"mean relative error: {mean_rel:.8f}")
    print(f"max relative error: {max_rel:.8f}")
    print(f"min relative error: {min_rel:.1f}")

    # 存储指标结果，type不同，文件不同
    if config.type == 1:
        result_file = "log/distnet_1_results_sample.txt"
    elif config.type == 2:
        result_file = "log/distnet_tilde_L1_results_sample.txt"
    elif config.type == 3:
        result_file = "log/distnet_L1_results_sample.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"Eval sample size: {len(trues)}\n")
        f.write(f"mean square error: {mse:.2f}\n")
        f.write(f"mean absolute error: {mae:.5f}\n")
        f.write(f"max absolute error: {max_abs:.4f}\n")
        f.write(f"min absolute error: {min_abs:.1f}\n")
        f.write(f"mean relative error: {mean_rel:.8f}\n")
        f.write(f"max relative error: {max_rel:.8f}\n")
        f.write(f"min relative error: {min_rel:.1f}\n")

if __name__ == "__main__":
    config, _ = get_config()
    eval_loader, maxLength, embed = load_and_preprocess_data(config, eval_sample_size=10000)
    test_model(eval_loader, config, maxLength, embed)

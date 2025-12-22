import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from distnet_model import ImprovedMultiLayerPerceptron
from config import get_config

class DistanceDataset(Dataset):
    def __init__(self, indices, node_features, sdm, LM_indices=None):
        self.indices = torch.from_numpy(indices).long()
        # 预先拼接特征，避免 __getitem__ 中的重复计算
        self.node_features = torch.from_numpy(node_features).float()
        self.sdm = torch.from_numpy(sdm).float()
        self.LM_indices = LM_indices if LM_indices else []

    def __len__(self):
        return len(self.indices) + len(self.LM_indices)

    def __getitem__(self, idx):
        if idx < len(self.indices):
            i, j = self.indices[idx]
        else:
            i, j = self.LM_indices[idx - len(self.indices)]
        return self.node_features[i], self.node_features[j], self.sdm[i, j]


def load_and_preprocess_data(config, city: str = "chengdu", data_root: str = "/home/lizhuoran/distance/data", eval_sample_size=10000):
    """
    Load and preprocess data for sampled evaluation
    :param config: Configuration parameters
    :param eval_sample_size: Number of samples for evaluation
    :return: sampled_eval_loader, maxLength
    """
    # Load preprocessed files from data/<city>/pre/ (fallback to data/pre)
    from pathlib import Path
    pre_dir = Path(data_root) / city / "pre"
    if not pre_dir.exists():
        pre_dir = Path(data_root) / "pre"

    sdm = np.load(pre_dir / "preprocessed_sdm.npy")
    embed = np.load(pre_dir / "preprocessed_embed.npy")
    node_long_lat = np.load(pre_dir / "preprocessed_node_long_lat.npy")
    node_long_lat_origin = np.load(pre_dir / "preprocessed_node_long_lat_origin.npy")

    # Print city information
    print(f"{city} with {sdm.shape[1]} nodes")

    # Normalize data
    # 修复：排除 inf 值后再计算最大距离
    maxLength = np.max(sdm[np.isfinite(sdm)])
    sdm = sdm / maxLength

    embed = np.array(embed)
    if embed.dtype == object:
        embed = np.array(list(embed))
    embed = (embed - embed.min()) / (embed.max() - embed.min())

    node_long_lat = np.array(node_long_lat)
    node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (
                node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
    node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (
                node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

    # 预融合特征
    node_features = np.concatenate((embed, node_long_lat), axis=1).astype(np.float32)

    # Create index pairs for non-zero and finite distances
    indices = np.argwhere((sdm != 0.0) & np.isfinite(sdm))

    # 随机采样全集中的若干合法节点对
    np.random.seed(123)
    sample_size = min(eval_sample_size, len(indices))
    sample_idxs = np.random.choice(len(indices), sample_size, replace=False)
    sampled_eval_indices = indices[sample_idxs]

    # Create dataset
    eval_dataset = DistanceDataset(sampled_eval_indices, node_features, sdm)

    # Create loader
    batch_size = config.batch_size
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return eval_loader, maxLength


def test_model(eval_loader, config, maxLength, city):
    """
    Test the pre-trained model and compute required metrics
    :param eval_loader: Test data loader
    :param config: Configuration parameters
    :param maxLength: scalar used to denormalize distances
    :param city: City name
    """
    # 自动检测设备 (支持 Mac MPS)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model initialization
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim

    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # Load pre-trained model (derive path from model_dir/save_prefix and city)
    import os
    model_dir = os.environ.get("DISTNET_MODEL_DIR", "param")
    save_prefix = os.environ.get("DISTNET_SAVE_PREFIX", "distnet_best")
    if os.getenv("DISTNET_MODEL_PATH"):
        model_path = os.getenv("DISTNET_MODEL_PATH")
    else:
        if config.type == 1:
            model_path = os.path.join(model_dir, f"{save_prefix}_{city}_1.ckpt")
        elif config.type == 2:
            model_path = os.path.join(model_dir, f"{save_prefix}_{city}_tilde_L1.ckpt")
        elif config.type == 3:
            model_path = os.path.join(model_dir, f"{save_prefix}_{city}_L1.ckpt")
        else:
            model_path = os.path.join(model_dir, f"{save_prefix}_{city}.ckpt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model: {model_path}")

    model.eval()  # Set model to evaluation mode

    # Collect predictions and truths (denormalized)
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in eval_loader:
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x1, batch_x2)
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

    # 存储指标结果，包含 city
    import os
    os.makedirs("log", exist_ok=True)
    if config.type == 1:
        result_file = f"log/{save_prefix}_{city}_results_sample_type1.txt"
    elif config.type == 2:
        result_file = f"log/{save_prefix}_{city}_results_sample_tilde_L1.txt"
    elif config.type == 3:
        result_file = f"log/{save_prefix}_{city}_results_sample_L1.txt"
    else:
        result_file = f"log/{save_prefix}_{city}_results_sample.txt"

    with open(result_file, 'w') as f:
        f.write(f"City: {city}\n")
        f.write(f"Eval sample size: {len(trues)}\n")
        f.write(f"mean square error: {mse:.2f}\n")
        f.write(f"mean absolute error: {mae:.5f}\n")
        f.write(f"max absolute error: {max_abs:.4f}\n")
        f.write(f"min absolute error: {min_abs:.1f}\n")
        f.write(f"mean relative error: {mean_rel:.8f}\n")
        f.write(f"max relative error: {max_rel:.8f}\n")
        f.write(f"min relative error: {min_rel:.1f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test DistNet on a city's preprocessed data sample")
    parser.add_argument("--city", type=str, default="chengdu", help="City name to test (folder under data/)")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory containing city folders")
    parser.add_argument("--model-dir", type=str, default="param", help="Directory where model checkpoints are stored")
    parser.add_argument("--save-prefix", type=str, default="distnet_best", help="Prefix used when saving checkpoints")
    parser.add_argument("--eval-samples", type=int, default=10000, help="Number of evaluation samples to draw")
    parser.add_argument("--model-path", type=str, default=None, help="If provided, load this exact checkpoint path instead of deriving from model-dir and city")
    # 使用 parse_known_args 以允许 config.py 处理它自己的参数
    args, _ = parser.parse_known_args()

    # export model path info to environment so load function can pick it up
    if args.model_path:
        import os
        os.environ["DISTNET_MODEL_PATH"] = args.model_path
    else:
        import os
        os.environ["DISTNET_MODEL_DIR"] = args.model_dir
        os.environ["DISTNET_SAVE_PREFIX"] = args.save_prefix

    config, _ = get_config()
    eval_loader, maxLength = load_and_preprocess_data(config, city=args.city, data_root=args.data_root, eval_sample_size=args.eval_samples)
    test_model(eval_loader, config, maxLength, city=args.city)

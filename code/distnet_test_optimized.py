import argparse
import numpy as np
import torch
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# 导入项目依赖
from distnet_model import ImprovedMultiLayerPerceptron
from config import get_config

class StaticTestDataset(Dataset):
    """
    用于测试的静态数据集
    """
    def __init__(self, indices, sdm, max_dist=1.0):
        self.indices = torch.from_numpy(indices).long()
        self.max_dist = max_dist
        # 提取标签并归一化
        self.targets = torch.from_numpy(sdm[indices[:, 0], indices[:, 1]] / max_dist).float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx, 0], self.indices[idx, 1], self.targets[idx]

def load_eval_data(config, city, data_root, eval_samples=10000):
    city_path = Path(data_root) / city / "pre"
    
    # 1. 加载特征 (必须与训练脚本 load_data_dynamic 逻辑完全一致)
    embed = np.load(city_path / "preprocessed_embed.npy").astype(np.float32)
    coords = np.load(city_path / "preprocessed_node_long_lat.npy").astype(np.float32)
    node_features = torch.from_numpy(np.hstack([embed, coords]))
    
    # 2. 加载 SDM (mmap 模式)
    sdm = np.load(city_path / "preprocessed_sdm.npy", mmap_mode='r')
    
    # 3. 计算 max_dist (必须与训练时一致)
    max_dist = np.max(sdm[np.isfinite(sdm)])
    print(f"Normalization factor (max_dist): {max_dist}")

    # 4. 随机采样测试对
    num_nodes = node_features.shape[0]
    test_indices = []
    print(f"Sampling {eval_samples} test pairs...")
    while len(test_indices) < eval_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and np.isfinite(sdm[u, v]):
            test_indices.append([u, v])
    
    test_indices = np.array(test_indices, dtype=np.int32)
    dataset = StaticTestDataset(test_indices, sdm, max_dist=max_dist)
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 1024), shuffle=False)
    
    return loader, node_features, max_dist

def evaluate(loader, node_features, max_dist, config, city, model_dir, save_prefix):
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    # 初始化模型架构 (必须与 distnet_train_optimized.py 一致)
    feature_dim = node_features.shape[1]
    hidden_dim = config.get('hidden_dim', 512)
    n_output = config.get('n_output', 64)
    
    model = ImprovedMultiLayerPerceptron(
        feature_dim * 2, hidden_dim, hidden_dim, hidden_dim, n_output
    ).to(device)

    # 构造模型路径
    model_type = config.get('type', 2)
    type_suffix = {1: "_1", 2: "_tilde_L1", 3: "_L1"}.get(model_type, "")
    model_path = Path(model_dir) / f"{save_prefix}_{city}{type_suffix}.ckpt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    node_features = node_features.to(device)
    all_preds = []
    all_targets = []
    all_preds_norm = []
    all_targets_norm = []

    start_time = time.time()
    with torch.no_grad():
        for u, v, target in loader:
            u, v, target = u.to(device), v.to(device), target.to(device)
            feat_u = node_features[u]
            feat_v = node_features[v]
            
            output = model(feat_u, feat_v).squeeze()
            
            # 保存原始归一化结果 (0-1 范围)
            all_preds_norm.append(output.cpu().numpy().reshape(-1))
            all_targets_norm.append(target.cpu().numpy().reshape(-1))

            # 反归一化：还原回真实距离
            all_preds.append(output.cpu().numpy().reshape(-1) * max_dist)
            all_targets.append(target.cpu().numpy().reshape(-1) * max_dist)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    preds_norm = np.concatenate(all_preds_norm)
    targets_norm = np.concatenate(all_targets_norm)

    # 计算归一化空间的指标
    abs_err_norm = np.abs(preds_norm - targets_norm)
    mae_norm = np.mean(abs_err_norm)
    mse_norm = np.mean((preds_norm - targets_norm) ** 2)
    max_abs_err_norm = np.max(abs_err_norm)
    min_abs_err_norm = np.min(abs_err_norm)

    # 计算相对误差 (0-1空间)
    mask_norm = targets_norm > 0
    rel_err_norm = abs_err_norm[mask_norm] / targets_norm[mask_norm]
    mre_norm = np.mean(rel_err_norm)
    max_rel_err_norm = np.max(rel_err_norm)
    min_rel_err_norm = np.min(rel_err_norm)

    print(f"\n--- Normalized Metrics (Raw Model Output 0-1) ---")
    print(f"mean square error: {mse_norm:.6f}")
    print(f"mean absolute error: {mae_norm:.6f}")
    print(f"max absolute error: {max_abs_err_norm:.6f}")
    print(f"min absolute error: {min_abs_err_norm:.6f}")
    print(f"mean relative error: {mre_norm:.6f}")
    print(f"max relative error: {max_rel_err_norm:.6f}")
    print(f"min relative error: {min_rel_err_norm:.6f}")
    
    # 计算指标
    abs_err = np.abs(preds - targets)
    mae = np.mean(abs_err)
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(abs_err)
    min_abs_err = np.min(abs_err)
    
    # 计算相对误差，避开 0 距离
    mask = targets > 0
    rel_err = abs_err[mask] / targets[mask]
    mre = np.mean(rel_err)
    max_rel_err = np.max(rel_err)
    min_rel_err = np.min(rel_err)

    print(f"\n--- Evaluation Results for {city} ---")
    print(f"Test Samples: {len(targets)}")
    print(f"mean square error: {mse:.2f}")
    print(f"mean absolute error: {mae:.5f}")
    print(f"max absolute error: {max_abs_err:.4f}")
    print(f"min absolute error: {min_abs_err:.1f}")
    print(f"mean relative error: {mre:.8f}")
    print(f"max relative error: {max_rel_err:.8f}")
    print(f"min relative error: {min_rel_err:.1f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    # 保存结果
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    result_file = log_dir / f"eval_{city}{type_suffix}.txt"
    with open(result_file, "w") as f:
        f.write(f"City: {city}\n")
        f.write(f"mean square error: {mse:.2f}\n")
        f.write(f"mean absolute error: {mae:.5f}\n")
        f.write(f"max absolute error: {max_abs_err:.4f}\n")
        f.write(f"min absolute error: {min_abs_err:.1f}\n")
        f.write(f"mean relative error: {mre:.8f}\n")
        f.write(f"max relative error: {max_rel_err:.8f}\n")
        f.write(f"min relative error: {min_rel_err:.1f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="beijing", help="City name")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--model-dir", default="./param")
    parser.add_argument("--save-prefix", default="distnet_best")
    parser.add_argument("--eval-samples", type=int, default=10000)
    args, _ = parser.parse_known_args()

    # 获取配置
    try:
        config, _ = get_config()
        if isinstance(config, argparse.Namespace):
            config = vars(config)
    except:
        config = {'batch_size': 1024, 'hidden_dim': 512, 'n_output': 64, 'type': 2}

    loader, node_features, max_dist = load_eval_data(
        config, args.city, args.data_root, eval_samples=args.eval_samples
    )
    
    evaluate(
        loader, node_features, max_dist, config, 
        args.city, args.model_dir, args.save_prefix
    )
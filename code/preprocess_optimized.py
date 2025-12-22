import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

# 简单的 Haversine 实现，避免依赖 distnet_model
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def process_city_optimized(data_dir, city, n_jobs=-1, generate_indices=False):
    """
    针对大规模节点优化的预处理函数 (SciPy 加速版)。
    1. 使用 scipy.sparse.csgraph 替代 networkx 计算最短路径 (速度提升 10x-50x)。
    2. 使用 float16 存储距离矩阵以节省内存。
    3. 支持并行分块计算。
    """
    city_dir = data_dir / city
    out_dir = city_dir / "pre"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Processing {city} (SciPy Accelerated) ===")
    
    # --- 1. 读取节点 ---
    node_file = city_dir / f"{city}_node-mod.txt"
    if not node_file.exists():
        node_file = city_dir / "nodes_lscc.txt"
        
    print(f"Loading nodes from {node_file}...")
    try:
        # 尝试读取，兼容有无header的情况
        df_nodes = pd.read_csv(node_file)
        if 'Node' in df_nodes.columns:
            node_ids = df_nodes['Node'].values
            coords = df_nodes[['Longitude', 'Latitude']].values
        else:
            # 无header fallback: 第一列ID，第二三列经纬度
            df_nodes = pd.read_csv(node_file, header=None)
            node_ids = df_nodes.iloc[:, 0].values
            coords = df_nodes.iloc[:, 1:3].values
    except Exception as e:
        print(f"Error reading nodes: {e}")
        return

    # 创建节点映射: original_id -> 0...N-1
    node_map = {old_id: i for i, old_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    print(f"Nodes: {num_nodes}")

    # 保存经纬度
    np.save(out_dir / "preprocessed_node_long_lat_origin.npy", coords.astype(np.float32))
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    np.save(out_dir / "preprocessed_node_long_lat.npy", coords_norm.astype(np.float32))

    # --- 2. 读取边并构建稀疏矩阵 (替换 NetworkX) ---
    edge_file = city_dir / f"{city}_link-mod.txt"
    if not edge_file.exists():
        edge_file = city_dir / "edges_lscc.txt"

    print(f"Loading edges from {edge_file}...")
    df_edges = pd.read_csv(edge_file)
    
    # 智能识别列名
    u_col, v_col, w_col = None, None, None
    possible_starts = ['Node_Start', 'Source', 'u', 'From']
    possible_ends = ['Node_End', 'Target', 'v', 'To']
    possible_weights = ['Length', 'Weight', 'w', 'dist']
    
    for col in df_edges.columns:
        if col in possible_starts: u_col = col
        if col in possible_ends: v_col = col
        if col in possible_weights: w_col = col
    
    if not (u_col and v_col):
        u_col, v_col = df_edges.columns[0], df_edges.columns[1]
        w_col = df_edges.columns[2] if len(df_edges.columns) > 2 else None

    # 映射边到 0..N-1
    # 过滤掉不在节点列表中的边
    valid_mask = df_edges[u_col].isin(node_map) & df_edges[v_col].isin(node_map)
    valid_edges = df_edges[valid_mask].copy()
    
    u_mapped = valid_edges[u_col].map(node_map).values
    v_mapped = valid_edges[v_col].map(node_map).values
    weights = valid_edges[w_col].values if w_col else np.ones(len(valid_edges))

    print(f"Building Sparse Matrix (Edges: {len(weights)})...")
    # 构建 CSR 稀疏矩阵
    graph_csr = sparse.csr_matrix((weights, (u_mapped, v_mapped)), 
                                  shape=(num_nodes, num_nodes))

    # --- 3. 计算 SDM (SciPy + Parallel) ---
    sdm_path = out_dir / "preprocessed_sdm.npy"
    print("Computing Shortest Distance Matrix (SDM) using SciPy...")
    
    start_time = time.time()
    
    # 预分配结果矩阵 (float16)
    # 注意：为了计算精度，我们先用 worker 返回 float32，最后存入 sdm
    sdm = np.full((num_nodes, num_nodes), np.inf, dtype=np.float16)

    # 定义分块大小
    chunk_size = 500  # 每个任务处理的行数，可根据内存调整
    node_indices = np.arange(num_nodes)
    chunks = [node_indices[i:i + chunk_size] for i in range(0, num_nodes, chunk_size)]

    def compute_chunk(indices):
        # SciPy shortest_path 是 C++ 实现，极快
        # directed=True, return_predecessors=False
        # indices 参数允许我们只计算部分源节点
        dists = shortest_path(graph_csr, method='auto', directed=True, 
                              return_predecessors=False, indices=indices)
        # 转换为 float32 以便传输，并替换 inf
        # 注意：scipy 返回 inf 表示不可达
        return dists.astype(np.float32)

    # 并行计算
    # n_jobs=-1 使用所有核心
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_chunk)(chunk) for chunk in tqdm(chunks, desc="Computing SDM")
    )
    
    # 组装矩阵
    print("Assembling SDM...")
    current_row = 0
    max_dist_check = 0
    
    for dist_chunk in results:
        rows = dist_chunk.shape[0]
        # 检查最大值，防止 float16 溢出
        chunk_max = dist_chunk[np.isfinite(dist_chunk)].max() if np.isfinite(dist_chunk).any() else 0
        if chunk_max > max_dist_check:
            max_dist_check = chunk_max
            
        sdm[current_row:current_row + rows, :] = dist_chunk
        current_row += rows

    print(f"SDM computed in {time.time() - start_time:.2f}s")
    
    if max_dist_check > 65500:
        print("Warning: Distances exceed float16 range (65504). Switching SDM to float32.")
        sdm = sdm.astype(np.float32)

    # 保存 SDM
    print(f"Saving SDM to {sdm_path}...")
    np.save(sdm_path, sdm)
    file_size = sdm_path.stat().st_size / (1024 * 1024)
    print(f"SDM saved. Size: {file_size:.2f} MB")

    # --- 3.5 生成地标 (Landmarks) ---
    # 移植自 preprocess_all.py / distnet_model
    print("Selecting Landmarks (Farthest Point Sampling)...")
    num_landmarks = max(int(num_nodes * 0.01), 20)
    # 使用原始坐标计算地标
    points = coords.astype(np.float32)
    
    # 简单的 FPS 实现
    landmarks_indices = [np.random.randint(num_nodes)]
    # 维护每个点到当前已选地标集合的最短距离
    min_dists = np.full(num_nodes, np.inf)
    
    for _ in tqdm(range(num_landmarks - 1), desc="Landmarks"):
        last_lm = points[landmarks_indices[-1]]
        # 更新距离: dist(p, last_lm)
        dists = haversine_np(points[:,0], points[:,1], last_lm[0], last_lm[1])
        min_dists = np.minimum(min_dists, dists)
        # 选择距离最大的点
        next_lm_idx = np.argmax(min_dists)
        landmarks_indices.append(next_lm_idx)
        
    lm_indices_path = out_dir / "preprocessed_LM_indices.npy"
    # 生成地标对索引 (只保留可达的)
    lm_pairs = [(i, j) for i in landmarks_indices for j in landmarks_indices if np.isfinite(sdm[i, j])]
    np.save(lm_indices_path, np.array(lm_pairs, dtype=np.int32))
    print(f"Landmarks saved: {len(lm_pairs)} pairs.")

    # --- 4. 处理 Indices (可选) ---
    indices_path = out_dir / "preprocessed_indices.npy"
    
    if generate_indices:
        print("Generating Indices (valid paths)...")
        # 分块处理 argwhere 以节省内存
        # 如果矩阵太大，直接 np.argwhere 会 OOM
        
        valid_pairs_count = np.isfinite(sdm).sum()
        print(f"Total reachable pairs: {valid_pairs_count}")
        
        if valid_pairs_count > 500_000_000:
            print("WARNING: Indices file will be huge (>4GB). Consider skipping.")
        
        # 我们可以用内存映射或者分块写入的方式
        # 这里为了简单，如果内存够用：
        try:
            indices = np.argwhere(np.isfinite(sdm)).astype(np.int32)
            print(f"Saving indices to {indices_path}...")
            np.save(indices_path, indices)
            idx_size = indices_path.stat().st_size / (1024 * 1024)
            print(f"Indices saved. Size: {idx_size:.2f} MB")
        except MemoryError:
            print("Error: Not enough memory to generate indices array at once.")
    else:
        print("Skipping indices.npy generation. (Optimized choice)")
        with open(out_dir / "indices_skipped.txt", "w") as f:
            f.write("Indices generation skipped for performance.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--city", default="beijing", type=str)
    parser.add_argument("--n-jobs", default=-1, type=int)
    parser.add_argument("--gen-indices", action="store_true", help="Force generation of indices.npy")
    args = parser.parse_args()
    
    process_city_optimized(Path(args.data_dir), args.city, n_jobs=args.n_jobs, generate_indices=args.gen_indices)

if __name__ == "__main__":
    main()
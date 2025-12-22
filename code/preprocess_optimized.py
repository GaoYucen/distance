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

# -----------------------------------------------------------------------------
# 简单的辅助函数，避免外部依赖
# -----------------------------------------------------------------------------

def haversine_np(lon1, lat1, lon2, lat2):
    """
    计算两点间的 Haversine 距离 (km)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def load_node2vec_manual(emb_path):
    """
    手动读取 node2vec .emb 文件 (Word2Vec 文本格式)，
    替代 gensim 以避免 scipy 版本冲突问题。
    格式通常为:
    <num_nodes> <dim>
    <node_id> <v1> <v2> ...
    """
    vectors = {}
    dim = 0
    with open(emb_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            # 正常带有 header 的情况
            count, dim = int(first_line[0]), int(first_line[1])
        else:
            # 无 header 的情况，回退指针
            f.seek(0)
            # 读取第一行判断维度 (id + vector)
            dim = len(first_line) - 1
            
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1: continue
            
            node_id = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vectors[node_id] = vec
            
            if dim == 0: dim = len(vec)
            
    return vectors, dim

# -----------------------------------------------------------------------------
# 核心处理逻辑
# -----------------------------------------------------------------------------

def process_city_optimized(data_dir, city, n_jobs=-1, generate_indices=False):
    """
    针对大规模节点优化的预处理函数 (SciPy 加速版)。
    1. 使用 scipy.sparse.csgraph 替代 networkx 计算最短路径。
    2. 使用 float16 存储距离矩阵以节省内存。
    3. 支持跳过已存在的 SDM 计算。
    """
    city_dir = data_dir / city
    out_dir = city_dir / "pre"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Processing {city} (SciPy Accelerated) ===")
    
    # --- 1. 读取节点 (始终执行，因为后续生成地标需要坐标) ---
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

    # --- Check: 检查 SDM 是否存在 ---
    sdm_path = out_dir / "preprocessed_sdm.npy"
    sdm = None

    if sdm_path.exists():
        print(f"✅ Found existing SDM at {sdm_path}. Loading directly...")
        # 直接加载，跳过边读取和计算
        sdm = np.load(sdm_path)
        print(f"SDM Loaded from disk. Shape: {sdm.shape}")
    else:
        print(f"SDM not found. Starting computation...")
        
        # --- 2. 读取边并构建稀疏矩阵 (仅在需要计算SDM时执行) ---
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
        valid_mask = df_edges[u_col].isin(node_map) & df_edges[v_col].isin(node_map)
        valid_edges = df_edges[valid_mask].copy()
        
        u_mapped = valid_edges[u_col].map(node_map).values
        v_mapped = valid_edges[v_col].map(node_map).values
        weights = valid_edges[w_col].values if w_col else np.ones(len(valid_edges))

        # --- [NEW] 保存处理后的边列表 (对齐 preprocess_all.py) ---
        edges_out_path = out_dir / "preprocessed_edges.csv"
        print(f"Saving remapped edges to {edges_out_path}...")
        pd.DataFrame({
            'u': u_mapped,
            'v': v_mapped,
            'weight': weights
        }).to_csv(edges_out_path, index=False)

        print(f"Building Sparse Matrix (Edges: {len(weights)})...")
        # 构建 CSR 稀疏矩阵
        graph_csr = sparse.csr_matrix((weights, (u_mapped, v_mapped)), 
                                      shape=(num_nodes, num_nodes))

        # --- 3. 计算 SDM (SciPy + Parallel) ---
        print("Computing Shortest Distance Matrix (SDM) using SciPy...")
        start_time = time.time()
        
        # 预分配结果矩阵 (float16)
        sdm = np.full((num_nodes, num_nodes), np.inf, dtype=np.float16)

        chunk_size = 500
        node_indices = np.arange(num_nodes)
        chunks = [node_indices[i:i + chunk_size] for i in range(0, num_nodes, chunk_size)]

        def compute_chunk(indices):
            dists = shortest_path(graph_csr, method='auto', directed=True, 
                                  return_predecessors=False, indices=indices)
            return dists.astype(np.float32)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_chunk)(chunk) for chunk in tqdm(chunks, desc="Computing SDM")
        )
        
        print("Assembling SDM...")
        current_row = 0
        max_dist_check = 0
        
        for dist_chunk in results:
            rows = dist_chunk.shape[0]
            chunk_max = dist_chunk[np.isfinite(dist_chunk)].max() if np.isfinite(dist_chunk).any() else 0
            if chunk_max > max_dist_check:
                max_dist_check = chunk_max
                
            sdm[current_row:current_row + rows, :] = dist_chunk
            current_row += rows

        print(f"SDM computed in {time.time() - start_time:.2f}s")
        
        if max_dist_check > 65500:
            print("Warning: Distances exceed float16 range (65504). Switching SDM to float32.")
            sdm = sdm.astype(np.float32)

        print(f"Saving SDM to {sdm_path}...")
        np.save(sdm_path, sdm)
        file_size = sdm_path.stat().st_size / (1024 * 1024)
        print(f"SDM saved. Size: {file_size:.2f} MB")

    # --- 3.5 生成地标 (需要 sdm 数据) ---
    print("Selecting Landmarks (Farthest Point Sampling)...")
    num_landmarks = max(int(num_nodes * 0.01), 20)
    points = coords.astype(np.float32)
    
    landmarks_indices = [np.random.randint(num_nodes)]
    min_dists = np.full(num_nodes, np.inf)
    
    for _ in tqdm(range(num_landmarks - 1), desc="Landmarks"):
        last_lm = points[landmarks_indices[-1]]
        dists = haversine_np(points[:,0], points[:,1], last_lm[0], last_lm[1])
        min_dists = np.minimum(min_dists, dists)
        next_lm_idx = np.argmax(min_dists)
        landmarks_indices.append(next_lm_idx)
        
    lm_indices_path = out_dir / "preprocessed_LM_indices.npy"
    
    # 过滤出可达的地标对
    # 注意：sdm 必须已加载或计算完成
    lm_pairs = []
    if sdm is not None:
        lm_pairs = [(i, j) for i in landmarks_indices for j in landmarks_indices if np.isfinite(sdm[i, j])]
        np.save(lm_indices_path, np.array(lm_pairs, dtype=np.int32))
        print(f"Landmarks saved: {len(lm_pairs)} pairs.")
    else:
        print("Error: SDM is None, cannot generate landmark pairs.")

    # --- 4. 处理 Indices ---
    indices_path = out_dir / "preprocessed_indices.npy"
    if generate_indices:
        if indices_path.exists():
             print(f"Indices file already exists at {indices_path}, skipping generation.")
        else:
            print("Generating Indices (valid paths)...")
            valid_pairs_count = np.isfinite(sdm).sum()
            print(f"Total reachable pairs: {valid_pairs_count}")
            
            if valid_pairs_count > 500_000_000:
                print("WARNING: Indices file will be huge (>4GB). Consider skipping.")
            
            try:
                indices = np.argwhere(np.isfinite(sdm)).astype(np.int32)
                print(f"Saving indices to {indices_path}...")
                np.save(indices_path, indices)
            except MemoryError:
                print("Error: Not enough memory to generate indices array at once.")
    else:
        print("Skipping indices.npy generation. (Optimized choice)")
        with open(out_dir / "indices_skipped.txt", "w") as f:
            f.write("Indices generation skipped for performance.")

def convert_emb_to_aligned_npy(data_dir, city):
    """
    根据已有的 .emb 文件生成对齐后的 preprocessed_embed.npy。
    使用手动加载替代 gensim。
    """
    data_path = Path(data_dir)
    city_dir = data_path / city
    out_dir = city_dir / "pre"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 [Alignment] Processing embeddings for {city}...")

    # 1. 读取节点列表以确定顺序
    node_file = city_dir / f"{city}_node-mod.txt"
    if not node_file.exists():
        node_file = city_dir / "nodes_lscc.txt"
    
    print(f"📖 Loading nodes to define index order: {node_file.name}")
    try:
        df_nodes = pd.read_csv(node_file)
        if 'Node' in df_nodes.columns:
            node_ids = df_nodes['Node'].values
        else:
            df_nodes = pd.read_csv(node_file, header=None)
            node_ids = df_nodes.iloc[:, 0].values
    except Exception as e:
        print(f"❌ Error reading nodes: {e}")
        return

    num_nodes = len(node_ids)
    node_to_idx = {str(node_id): i for i, node_id in enumerate(node_ids)}
    
    # 2. 读取 Embedding 文件
    emb_file = city_dir / f"{city}_node2vec.emb"
    if not emb_file.exists():
        print(f"❌ Error: Embedding file not found at {emb_file}")
        return

    print(f"📖 Loading external embedding file (Manual): {emb_file.name}")
    
    try:
        # 使用手动加载函数，移除 gensim 依赖
        vectors_map, vector_dim = load_node2vec_manual(emb_file)
        
        print(f"✅ Embedding dimension: {vector_dim}")
        print("⚡ Aligning embeddings to node index...")
        
        embed_matrix = np.zeros((num_nodes, vector_dim), dtype=np.float32)
        
        found_count = 0
        for node_id_str, idx in node_to_idx.items():
            if node_id_str in vectors_map:
                embed_matrix[idx] = vectors_map[node_id_str]
                found_count += 1
        
        print(f"📊 Matched nodes: {found_count}/{num_nodes} ({found_count/num_nodes:.1%})")

        output_path = out_dir / "preprocessed_embed.npy"
        np.save(output_path, embed_matrix)
        print(f"✅ Successfully saved aligned matrix to: {output_path}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--city", default="beijing", type=str)
    parser.add_argument("--n-jobs", default=-1, type=int)
    parser.add_argument("--gen-indices", action="store_true", help="Force generation of indices.npy")
    args = parser.parse_args()
    
    process_city_optimized(Path(args.data_dir), args.city, n_jobs=args.n_jobs, generate_indices=args.gen_indices)
    convert_emb_to_aligned_npy(args.data_dir, args.city)

if __name__ == "__main__":
    main()
import os
import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from unittest.mock import MagicMock

# --- Fix for SciPy compatibility (Environment Patching) ---
# 错误: "C function scipy.spatial._qhull._barycentric_coordinates has wrong signature"
# 原因: Scipy 环境文件损坏或版本冲突 (常见于 Python 3.11 + Scipy < 1.11)
# 方案: 尝试导入 scipy.interpolate，如果崩溃，则将其 Mock 掉。
#       Node2Vec/Gensim 的核心训练逻辑不需要插值功能，因此可以安全跳过。
try:
    import scipy.interpolate
except Exception as e:
    print(f"⚠️ Warning: Detected broken scipy.interpolate ({e}).")
    print("⚠️ Patching scipy.interpolate with MagicMock to allow Gensim to load...")
    sys.modules['scipy.interpolate'] = MagicMock()

# --- Fix for SciPy linalg compatibility ---
import scipy
import scipy.linalg
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu
if not hasattr(scipy.linalg, 'tril'):
    scipy.linalg.tril = np.tril

from scipy import sparse
from gensim.models import Word2Vec

# [FIXED] Correct import for PecanPy v2.x
from pecanpy.pecanpy import SparseOTF, DenseOTF

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def run_node2vec_optimized(data_dir, city, output_dir, embedding_dim=64, walk_length=80, num_walks=10, p=1, q=1, workers=-1):
    """
    优化版的 Node2Vec 流程：
    1. 使用 Pandas + Scipy 直接构建稀疏矩阵 (CSR)，跳过 NetworkX。
    2. 直接调用 PecanPy 的 from_mat 接口。
    3. 并行生成游走并训练。
    """
    if workers < 1:
        workers = multiprocessing.cpu_count()
        
    print(f"\n🚀 [Optimized] Processing {city} with {workers} workers...")
    
    city_dir = os.path.join(data_dir, city)
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. 快速读取节点 (构建映射)
    # ---------------------------------------------------------
    node_file = os.path.join(city_dir, f"{city}_node-mod.txt")
    if not os.path.exists(node_file):
        node_file = os.path.join(city_dir, "nodes_lscc.txt") # Fallback
        
    print(f"📖 Loading nodes from {os.path.basename(node_file)}...")
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

    # 构建映射表: Original ID -> Matrix Index (0..N-1)
    # 这比 NetworkX 节点查找快得多
    node_map = {old_id: i for i, old_id in enumerate(node_ids)}
    index_map = {i: old_id for i, old_id in enumerate(node_ids)} # 反向映射用于保存
    num_nodes = len(node_ids)
    print(f"✅ Nodes: {num_nodes}")

    # ---------------------------------------------------------
    # 2. 快速读取边并构建 CSR 矩阵 (关键优化点)
    # ---------------------------------------------------------
    edge_file = os.path.join(city_dir, f"{city}_link-mod.txt")
    if not os.path.exists(edge_file):
        edge_file = os.path.join(city_dir, "edges_lscc.txt")
        
    print(f"📖 Loading edges from {os.path.basename(edge_file)}...")
    df_edges = pd.read_csv(edge_file)
    
    # 自动识别列
    u_col, v_col, w_col = None, None, None
    possible_starts = ['Node_Start', 'Source', 'u', 'From']
    possible_ends = ['Node_End', 'Target', 'v', 'To']
    possible_weights = ['Length', 'Weight', 'w', 'dist']
    
    cols = df_edges.columns
    for c in cols:
        if c in possible_starts: u_col = c
        elif c in possible_ends: v_col = c
        elif c in possible_weights: w_col = c
        
    # Fallback to indices if names not found
    if not u_col: u_col = cols[0]
    if not v_col: v_col = cols[1]
    
    # 过滤无效边并映射 ID
    valid_mask = df_edges[u_col].isin(node_map) & df_edges[v_col].isin(node_map)
    valid_edges = df_edges[valid_mask]
    
    u_mapped = valid_edges[u_col].map(node_map).values
    v_mapped = valid_edges[v_col].map(node_map).values
    
    if w_col:
        weights = valid_edges[w_col].values
        # Node2Vec 需要相似度权重而不是距离权重吗？
        # 通常道路网络权重是距离。Node2Vec 处理时，距离越短权重越小？
        # PecanPy/Node2Vec 将权重视为转移概率的未归一化值。
        # 如果权重是距离(Length)，需要取倒数或者转换，否则游走会倾向于走长边。
        # 这里假设保持原值，或者您可以取消注释下面这行：
        # weights = 1.0 / (weights + 1e-6) 
    else:
        weights = np.ones(len(u_mapped), dtype=np.float32)

    print("⚡ Building Sparse CSR Matrix directly...")
    adj_mat = sparse.csr_matrix((weights, (u_mapped, v_mapped)), 
                                shape=(num_nodes, num_nodes))
    
    # ---------------------------------------------------------
    # 3. 使用 PecanPy 生成随机游走
    # ---------------------------------------------------------
    print(f"🧠 Initializing PecanPy (SparseOTF, p={p}, q={q})...")
    
    # 对于 10k+ 节点，始终使用 SparseOTF 以节省内存并保持高效
    # extend=False 表示不包含隐式自环，通常路网不需要
    g = SparseOTF(p=p, q=q, workers=workers, verbose=True, extend=False)
    
    # 直接从矩阵加载，无需 NetworkX 转换，速度极快
    # directed=True 适用于路网
    g.from_mat(adj_mat, weighted=True, directed=True)
    
    print(f"🚶 Generating walks (Num={num_walks}, Len={walk_length})...")
    # simulate_walks 返回的是 uint32 的 numpy array 或 list
    walks_raw = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)
    
    # ---------------------------------------------------------
    # 4. 训练 Word2Vec
    # ---------------------------------------------------------
    print("🔄 Converting walks to strings for Gensim...")
    # Gensim 需要字符串列表。这一步可能是内存瓶颈，尽量优化。
    # walks_raw 是 [num_nodes * num_walks, walk_length] 的矩阵
    
    # 使用生成器或列表推导式转换
    # 注意：PecanPy 返回的是 0..N-1 的索引，我们需要训练这些索引
    walks = [[str(node) for node in walk] for walk in walks_raw]
    
    print(f"🏋️ Training Word2Vec (Dim={embedding_dim})...")
    model = Word2Vec(walks, 
                     vector_size=embedding_dim, 
                     window=10, 
                     min_count=0, 
                     sg=1, 
                     workers=workers, 
                     epochs=1)

    # ---------------------------------------------------------
    # 5. 保存结果
    # ---------------------------------------------------------
    output_model_file = os.path.join(output_dir, f"{city}_node2vec.model")
    output_embed_file = os.path.join(output_dir, "preprocessed_embed.npy") # 保存为字典 pickle
    
    print(f"💾 Saving model to {output_model_file}...")
    model.save(output_model_file)
    
    print("💾 Extracting and remapping embeddings...")
    embeddings = {}
    # 将 0..N-1 的索引映射回原始 ID (如 30905)
    for idx in range(num_nodes):
        str_idx = str(idx)
        if str_idx in model.wv:
            original_id = index_map[idx]
            embeddings[original_id] = model.wv[str_idx]
            
    with open(output_embed_file, 'wb') as f:
        pickle.dump(embeddings, f)
        
    print(f"✅ Done! Embeddings saved to {output_embed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--city", default="beijing", type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n-jobs", default=-1, type=int)
    
    args = parser.parse_args()
    
    out_dir = args.output_dir if args.output_dir else os.path.join(args.data_dir, args.city, "pre")
    
    run_node2vec_optimized(
        data_dir=args.data_dir,
        city=args.city,
        output_dir=out_dir,
        embedding_dim=args.dim,
        workers=args.n_jobs
    )
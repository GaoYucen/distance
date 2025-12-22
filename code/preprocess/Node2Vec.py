#%%
import pickle
import numpy as np
import pandas as pd
import sys
import argparse
import os
import logging
import multiprocessing

# --- Fix for SciPy 1.11+ compatibility with gensim ---
import scipy
import scipy.sparse
import scipy.linalg

# 针对 SciPy 1.13+ 移除 triu/tril 的兼容性修复
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu
if not hasattr(scipy.linalg, 'tril'):
    scipy.linalg.tril = np.tril

import networkx as nx 
from pecanpy import pecanpy 
from gensim.models import Word2Vec
from tqdm import tqdm

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- 路径与配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.abspath(os.path.join(current_dir, ".."))
if code_dir not in sys.path:
    sys.path.append(code_dir)

# 尝试导入配置，如果失败则使用默认值（方便单文件调试）
try:
    from config import get_config
    params, _ = get_config()
    EMBED_DIM = params.embed_dim
except ImportError:
    print("⚠️ 警告: 未找到 config.py，使用默认配置。")
    class Config:
        pass
    EMBED_DIM = 64  # 默认维度

# --- 常量定义 ---
DATA_DIR_TEMPLATE = './data/{city}/'
EMBED_FILE_TEMPLATE = '{city}_node2vec_embed.pkl'
MODEL_FILE_TEMPLATE = '{city}_node2vec.emb'

def ensure_csr_matrix(adj):
    """
    确保邻接矩阵是 scipy.sparse.csr_matrix 格式。
    处理 NetworkX 版本差异导致的 csr_array vs csr_matrix 问题。
    """
    if scipy.sparse.isspmatrix_csr(adj):
        return adj
    try:
        # 尝试转换（适用于 csr_array）
        return scipy.sparse.csr_matrix(adj)
    except Exception as e:
        print(f"❌ 矩阵转换失败: {e}")
        return adj

def generate_node_embeddings(city_name):
    print(f"\n--- 🚀 开始处理城市 Node2Vec 嵌入: {city_name.upper()} ---")
    
    city_data_dir = DATA_DIR_TEMPLATE.format(city=city_name)
    output_model_file = os.path.join(city_data_dir, MODEL_FILE_TEMPLATE.format(city=city_name))
    output_embed_file = os.path.join(city_data_dir, EMBED_FILE_TEMPLATE.format(city=city_name))
    
    # 1. 优先读取 LSCC 文件，否则读取普通文件
    edge_file = os.path.join(city_data_dir, 'edges_lscc.txt')
    if not os.path.exists(edge_file):
        print(f"ℹ️ 未找到 edges_lscc.txt，尝试读取 edges.txt")
        edge_file = os.path.join(city_data_dir, 'edges.txt')
    
    if not os.path.exists(edge_file):
        print(f"❌ 错误: 找不到边文件 {edge_file}")
        return

    try:
        print(f"Reading edges from {edge_file}...")
        # 假设 convert_graphs.py 生成的标准格式: Node_Start, Node_End, Length
        df_edges = pd.read_csv(edge_file)
        
        G = nx.DiGraph()
        # 批量添加边 (u, v, weight)
        edges_to_add = [(int(row['Node_Start']), int(row['Node_End']), float(row['Length'])) for _, row in df_edges.iterrows()]
        G.add_weighted_edges_from(edges_to_add)
        print(f"✅ 图构建完成。节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    except Exception as e:
        print(f"❌ 构建图时发生错误: {e}")
        return

    # --- 1.5 🎯 Node2Vec 预处理 ---
    
    # 移除无效权重边
    remove_edges = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0) # 默认为1
        if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            remove_edges.append((u, v))

    if remove_edges:
        print(f"⚠️ 移除 {len(remove_edges)} 条无效权重的边。")
        G.remove_edges_from(remove_edges)

    # 处理有向图的出度为0节点
    if G.is_directed():
        no_out_degree_nodes = [node for node, degree in G.out_degree() if degree == 0]
        if no_out_degree_nodes:
            print(f"⚠️ [有向图] 移除 {len(no_out_degree_nodes)} 个出度为 0 的节点。")
            G.remove_nodes_from(no_out_degree_nodes)
    
    # 连通性检查 (兼容有向和无向图)
    if G.number_of_nodes() > 0:
        if G.is_directed():
            if not nx.is_strongly_connected(G):
                print('⚠️ 图非强连通，提取最大强连通分量 (SCC)...')
                scs = list(nx.strongly_connected_components(G))
                max_sc = max(scs, key=len)
                G = G.subgraph(max_sc).copy()
        else:
            if not nx.is_connected(G):
                print('⚠️ 图非连通，提取最大连通分量 (CC)...')
                ccs = list(nx.connected_components(G))
                max_cc = max(ccs, key=len)
                G = G.subgraph(max_cc).copy()
                
    if G.number_of_nodes() == 0:
        print("❌ 警告: 清理后图为空，跳过。")
        return

    print(f"✅ 预处理完成。最终图规模：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    
    # 2. 生成 Embedding
    print("⏳ 开始运行 Node2Vec (PecanPy)...")
    
    WALK_LEN = 20
    NUM_WALKS = 80
    
    # 确保节点列表排序
    node_list_orig = sorted(list(G.nodes()))
    node_list_str = [str(n) for n in node_list_orig]
    
    # --- 关键修复：矩阵转换 ---
    # NetworkX 2.7+ 推荐用 to_scipy_sparse_array，旧版用 to_scipy_sparse_matrix
    # 这里做个兼容处理
    try:
        if hasattr(nx, 'to_scipy_sparse_array'):
            adj_mat = nx.to_scipy_sparse_array(G, nodelist=node_list_orig, weight='weight', format='csr')
        else:
            adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=node_list_orig, weight='weight', format='csr')
            
        # 强制转换为 csr_matrix (PecanPy 可能不支持 csr_array)
        adj_mat = ensure_csr_matrix(adj_mat)
        
    except Exception as e:
        print(f"❌ 邻接矩阵转换失败: {e}")
        return

    # 初始化 PecanPy
    walks = None
    
    # 策略: 优先尝试 SparseOTF，如果因为 SciPy 版本兼容性报错，则自动切换到 DenseOTF
    try:
        print("   -> 尝试模式: SparseOTF")
        p_g = pecanpy.SparseOTF.from_mat(adj_mat, node_list_str, p=1, q=1, workers=-1, verbose=True)
        walks = p_g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LEN)
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ SparseOTF 运行遇到问题: {error_msg}")
        
        # 检测是否为 "ambiguous truth value" 错误 (SciPy/NumPy 版本冲突常见错误)
        if "truth value of an array" in error_msg or "ambiguous" in error_msg:
            print("   -> 💡 检测到 SciPy 版本兼容性问题，正在切换到 DenseOTF (稠密矩阵) 模式...")
            print("   -> 该模式对中小规模图 (节点<10000) 非常有效且稳定。")
            
            try:
                # 转换为稠密矩阵 (numpy array)
                adj_dense = adj_mat.toarray()
                # 使用 DenseOTF
                p_g = pecanpy.DenseOTF.from_mat(adj_dense, node_list_str, p=1, q=1, workers=-1, verbose=True)
                walks = p_g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LEN)
            except Exception as e2:
                print(f"❌ DenseOTF 也运行失败: {e2}")
                return
        else:
            print("❌ 无法自动恢复的错误，停止运行。")
            return

    if walks is None:
        print("❌ 随机游走生成失败。")
        return
    
    # Gensim 训练
    cores = multiprocessing.cpu_count()
    print(f"⏳ 开始训练 Word2Vec (Threads={cores})...")
    
    model = Word2Vec(walks, vector_size=EMBED_DIM, window=10, min_count=1, 
                     sg=1, workers=cores, epochs=1)

    # 3. 保存模型
    model.wv.save_word2vec_format(output_model_file)
    print(f"✅ 模型已保存: {output_model_file}")

    # 4. 提取并保存字典格式嵌入
    embeddings = {}
    print("⏳ 提取嵌入向量...")
    # 直接使用 KeyedVectors 遍历更高效
    for node_id in node_list_str:
        if node_id in model.wv:
            # 尝试转回原始类型（如果原图节点是int）
            try:
                original_key = int(node_id)
            except ValueError:
                original_key = node_id
            
            # 只有当原始图里有这个节点时才保存（虽然通常是一致的）
            if original_key in G.nodes():
                embeddings[original_key] = model.wv[node_id]

    with open(output_embed_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✅ 嵌入字典已保存: {output_embed_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Node2Vec embeddings for road networks.')
    parser.add_argument('--cities', nargs='+', default=['chengdu'], help='List of city names to process.')
    args = parser.parse_args()

    for city in args.cities:
        generate_node_embeddings(city)
    print("\n--- ✅ 处理完毕 ---")
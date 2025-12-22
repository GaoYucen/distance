#%%
import numpy as np
import pandas as pd
import sys
import argparse
import os
import logging
import multiprocessing
import math
import pickle

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

# 尝试导入配置
try:
    from config import get_config
    params, _ = get_config()
    EMBED_DIM = params.embed_dim
except ImportError:
    print("⚠️ 警告: 未找到 config.py，使用默认配置。")
    EMBED_DIM = 64

# --- 配置 ---
# --- 核心辅助函数 ---

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点间的 Haversine 距离 (单位: km)
    """
    R = 6371  # 地球半径 (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def ensure_csr_matrix(adj):
    """确保矩阵为 scipy.sparse.csr_matrix 格式"""
    if scipy.sparse.isspmatrix_csr(adj):
        return adj
    try:
        return scipy.sparse.csr_matrix(adj)
    except Exception as e:
        print(f"❌ 矩阵转换失败: {e}")
        return adj

# --- 主逻辑 ---

def process_node2vec_haversine(city_name):
    print(f"\n--- 🚀 开始处理 Node2Vec (Haversine 加权): {city_name.upper()} ---")
    
    data_dir = f'./data/{city_name}/'
    embed_file = os.path.join(data_dir, f'{city_name}_node2vec_haversine_embed.pkl')
    model_file = os.path.join(data_dir, f'{city_name}_node2vec_haversine.emb')

    nodes_file = os.path.join(data_dir, 'nodes.txt')
    
    # 1. 优先读取 LSCC 文件，否则读取普通文件 (参考 Node2Vec.py)
    edge_file = os.path.join(data_dir, 'edges_lscc.txt')
    if not os.path.exists(edge_file):
        print(f"ℹ️ 未找到 edges_lscc.txt，尝试读取 edges.txt")
        edge_file = os.path.join(data_dir, 'edges.txt')
    
    if not os.path.exists(nodes_file) or not os.path.exists(edge_file):
        print(f"❌ 错误: 找不到数据文件。\n请检查 {data_dir} 下是否存在 nodes.txt 和 edges.txt")
        return

    # 2. 加载节点数据 (为了获取经纬度)
    print(f"⏳ 加载节点数据: {nodes_file}")
    node_coords = {}
    try:
        nodes_df = pd.read_csv(nodes_file)
        nodes_df.columns = [c.lower().strip() for c in nodes_df.columns]
        
        # 查找关键列
        id_col = next((c for c in ['osmid', 'id', 'node_id', 'node'] if c in nodes_df.columns), None)
        x_col = next((c for c in ['x', 'lon', 'longitude'] if c in nodes_df.columns), None)
        y_col = next((c for c in ['y', 'lat', 'latitude'] if c in nodes_df.columns), None)
        
        if not (id_col and x_col and y_col):
            raise ValueError(f"❌ 无法识别 CSV 列名。\nNodes cols: {nodes_df.columns}")
            
        for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Nodes"):
            node_coords[int(row[id_col])] = {'x': row[x_col], 'y': row[y_col]}
            
    except Exception as e:
        print(f"❌ 节点加载失败: {e}")
        return

    # 3. 构建图 (参考 Node2Vec.py 读取逻辑)
    print(f"⏳ 加载边数据: {edge_file}")
    try:
        df_edges = pd.read_csv(edge_file)
        G = nx.DiGraph()
        
        # 假设标准格式: Node_Start, Node_End
        edges_to_add = [(int(row['Node_Start']), int(row['Node_End'])) for _, row in df_edges.iterrows()]
        G.add_edges_from(edges_to_add)
        
        print(f"✅ 原始图构建完成。节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    except Exception as e:
        print(f"❌ 图构建失败: {e}")
        return

    # 4. 计算 Haversine 距离并调整权重
    # 逻辑: 距离越近，权重应该越大 (Node2Vec 倾向于游走到权重大的边)
    # 或者: 距离越远，作为"阻力"越大？
    # 通常在路网分析中，如果表示连通性，权重 = 1/距离 或者 exp(-距离)
    # 这里我们使用常见的权重调整：Weight = 1.0 / (Distance_km + epsilon)
    # 或者如果原意是用距离作为 cost，那在 Node2Vec 里通常需要反转为相似度/概率
    
    print("⏳ 计算 Haversine 距离并更新边权重...")
    count_updates = 0
    epsilon = 1e-5 # 防止除零
    
    for u, v in tqdm(G.edges(), desc="Updating Weights"):
        # 获取节点坐标
        try:
            # 确保节点存在且有坐标 (如果是从 edge 文件读入的节点但 node 文件里没有，可能会报错)
            if u not in node_coords or v not in node_coords:
                # 缺失坐标，赋予默认权重
                G[u][v]['weight'] = 1.0
                continue
                
            n1 = node_coords[u]
            n2 = node_coords[v]
                
            dist_km = haversine(n1['y'], n1['x'], n2['y'], n2['x'])
            
            # ⚠️ 关键决策：权重如何定义？
            # Node2Vec 的转移概率正比于权重。
            # 如果希望更倾向于走“短距离”的边（即地理上更紧密），权重应与距离成反比。
            weight = 1.0 / (dist_km + epsilon)
            
            # 更新图权重
            G[u][v]['weight'] = weight
            count_updates += 1
            
        except Exception as e:
            pass # 忽略个别数据错误

    print(f"✅ 已更新 {count_updates} 条边的权重。")

    # 3. 图清理 (移除孤立点，取最大连通分量)
    if G.is_directed():
        if not nx.is_strongly_connected(G):
            print('⚠️ 提取最大强连通分量...')
            scs = list(nx.strongly_connected_components(G))
            max_sc = max(scs, key=len)
            G = G.subgraph(max_sc).copy()
    else:
        if not nx.is_connected(G):
            print('⚠️ 提取最大连通分量...')
            ccs = list(nx.connected_components(G))
            max_cc = max(ccs, key=len)
            G = G.subgraph(max_cc).copy()
            
    print(f"✅ 预处理后图规模: 节点 {G.number_of_nodes()}, 边 {G.number_of_edges()}")

    # 4. 准备矩阵 (PecanPy)
    print("⏳ 准备邻接矩阵...")
    node_list_orig = sorted(list(G.nodes()))
    node_list_str = [str(n) for n in node_list_orig]
    
    try:
        # 兼容性处理
        if hasattr(nx, 'to_scipy_sparse_array'):
            adj_mat = nx.to_scipy_sparse_array(G, nodelist=node_list_orig, weight='weight', format='csr')
        else:
            adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=node_list_orig, weight='weight', format='csr')
        
        adj_mat = ensure_csr_matrix(adj_mat)
    except Exception as e:
        print(f"❌ 矩阵转换失败: {e}")
        return

    # 5. 运行 PecanPy (带降级重试机制)
    WALK_LEN = 20
    NUM_WALKS = 80
    walks = None
    
    try:
        print("⏳ 尝试 SparseOTF 模式...")
        p_g = pecanpy.SparseOTF.from_mat(adj_mat, node_list_str, p=1, q=1, workers=-1, verbose=True)
        walks = p_g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LEN)
    except Exception as e:
        print(f"⚠️ SparseOTF 失败: {e}")
        if "truth value" in str(e) or "ambiguous" in str(e):
            print("   -> 💡 切换至 DenseOTF (稠密) 模式...")
            try:
                adj_dense = adj_mat.toarray()
                p_g = pecanpy.DenseOTF.from_mat(adj_dense, node_list_str, p=1, q=1, workers=-1, verbose=True)
                walks = p_g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LEN)
            except Exception as e2:
                print(f"❌ DenseOTF 也失败: {e2}")
                return
        else:
            return

    if not walks:
        return

    # 6. 训练 Word2Vec
    cores = multiprocessing.cpu_count()
    print(f"⏳ 训练 Word2Vec (Cores={cores})...")
    model = Word2Vec(walks, vector_size=EMBED_DIM, window=10, min_count=1, 
                     sg=1, workers=cores, epochs=1)
    
    model.wv.save_word2vec_format(model_file)
    print(f"✅ 模型已保存: {model_file}")

    # 7. 提取并保存字典
    embeddings = {}
    for node_id in node_list_str:
        if node_id in model.wv:
            # 尝试转回原始类型
            try:
                original_key = int(node_id)
            except ValueError:
                original_key = node_id
            
            if original_key in G.nodes():
                embeddings[original_key] = model.wv[node_id]

    with open(embed_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✅ 嵌入结果已保存: {embed_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Haversine-weighted Node2Vec embeddings.')
    parser.add_argument('--cities', nargs='+', default=['chengdu'], help='List of city names to process.')
    args = parser.parse_args()

    for city in args.cities:
        process_node2vec_haversine(city)
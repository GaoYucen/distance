#%%
import pickle
import numpy as np
import pandas as pd
import sys
import os
import networkx as nx # 引入 networkx 用于图操作
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm

# 添加路径并获取配置
sys.path.append('/home/lizhuoran/distance/code')
from config import get_config

params, _ = get_config()

# --- 配置部分 ---
# 定义要处理的城市列表
CITIES = ['harbin', 'porto', 'india']
# 假设数据文件的基础路径结构
DATA_DIR_TEMPLATE = '/home/lizhuoran/distance/data/{city}/'
# 输入文件：图文件
GRAPH_FILE_TEMPLATE = '{city}_graph_sc.pkl'
# 输出文件模板
EMBED_FILE_TEMPLATE = '{city}_node2vec_embed.pkl'
MODEL_FILE_TEMPLATE = '{city}_node2vec.emb'

# --- 主处理函数 ---

def generate_node_embeddings(city_name):
    """
    加载指定城市的图文件，进行权重和出度过滤，使用 Node2Vec 生成节点嵌入。
    """
    print(f"\n--- 🚀 开始处理城市 Node2Vec 嵌入: {city_name.upper()} ---")
    
    # 构造城市数据文件夹路径
    city_data_dir = DATA_DIR_TEMPLATE.format(city=city_name)
    
    # 构造输入/输出文件路径
    graph_file = os.path.join(city_data_dir, GRAPH_FILE_TEMPLATE.format(city=city_name))
    output_model_file = os.path.join(city_data_dir, MODEL_FILE_TEMPLATE.format(city=city_name))
    output_embed_file = os.path.join(city_data_dir, EMBED_FILE_TEMPLATE.format(city=city_name))
    
    # 1. 读取图文件
    try:
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        print(f"✅ 成功加载图文件: {graph_file}")
        print(f"   初始图信息：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到图文件。请确保前一步已生成 {graph_file}")
        return
    except Exception as e:
        print(f"❌ 读取图文件时发生错误: {e}")
        return

    # --- 1.5 🎯 Node2Vec 容错性过滤步骤 ---
    
    # 移除权重为非正数或非有限值的边
    remove_edges = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight')
        
        # 检查权重是否为非有限值（NaN, Inf）或 0 或负数
        if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            remove_edges.append((u, v))

    if remove_edges:
        print(f"⚠️ 移除 {len(remove_edges)} 条权重为非正或非有限的边。")
        G.remove_edges_from(remove_edges)
    else:
        print("✅ 图中没有发现权重为非正或非有限的边。")

    # 移除出度为 0 的节点（Node2Vec 无法从这些节点开始游走）
    no_out_degree_nodes = [node for node, degree in G.out_degree() if degree == 0]

    if no_out_degree_nodes:
        print(f"⚠️ 移除 {len(no_out_degree_nodes)} 个出度为 0 的节点。")
        G.remove_nodes_from(no_out_degree_nodes)
    else:
        print("✅ 图中没有发现出度为 0 的节点。")
        
    # 重新检查图的连通性并提取最大 SCC（可选，但推荐）
    if G.number_of_nodes() > 0 and not nx.is_strongly_connected(G):
        print('⚠️ 过滤后图不再是强连通的，正在重新提取最大强连通分量...')
        scs = list(nx.strongly_connected_components(G))
        max_sc = max(scs, key=len)
        G = G.subgraph(max_sc).copy()
        print(f'✅ 最大 SCC 重新提取完成。Node2Vec 将运行在新图上：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}')
    elif G.number_of_nodes() == 0:
        print("❌ 警告: 清理后图为空，跳过 Node2Vec。")
        return
    else:
        print(f"✅ 过滤后图保持强连通。Node2Vec 将运行在新图上：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    
    # 2. 用Node2Vec算法生成节点embedding
    print("⏳ 开始运行 Node2Vec...")
    
    # 为了增加稳定性，将 workers 减少到 2 (如果您机器资源有限，可以考虑)
    # 原始: workers=4
    node2vec = Node2Vec(G, dimensions=params.embed_dim, walk_length=30, num_walks=200, workers=2) 
    
    # 运行模型
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print("✅ Node2Vec 模型训练完成。")

    # 3. 保存 Node2Vec 模型 (KeyedVectors)
    model.wv.save_word2vec_format(output_model_file)
    print(f"✅ Node2Vec 模型已保存到: {output_model_file}")

    # 4. 提取并保存嵌入结果
    # ... (这部分代码保持不变) ...
    node_list = list(G.nodes())
    node_list.sort()

    embeddings = {}
    print(f"⏳ 提取 {len(node_list)} 个节点的嵌入向量...")
    for node in tqdm(node_list, desc="提取嵌入"):
        # Node2Vec 模型将节点ID转换为字符串作为 key
        # 使用 model.wv 访问 KeyedVectors（Word2Vec 对象本身不可下标）
        embeddings[node] = model.wv[str(node)]

    # 5. 保存嵌入结果 (pickle)
    with open(output_embed_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✅ 嵌入结果已保存到: {output_embed_file}")


# --- 脚本执行入口 ---

if __name__ == '__main__':
    # 遍历所有城市并处理
    for city in CITIES:
        generate_node_embeddings(city)

    print("\n--- ✅ 所有城市 Node2Vec 嵌入处理完毕 (end) ---")
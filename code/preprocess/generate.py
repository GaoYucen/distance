import pandas as pd
import pickle
import networkx as nx
import numpy as np
import sys
from tqdm import tqdm
import os
# 导入配置的代码保持不变，如果不需要可以删除
# sys.path.append('code')
# from config import get_config
# params, _ = get_config()

# --- 配置部分 ---
# 定义要处理的城市列表
# CITIES = ['harbin', 'porto'] # 您可以改为 ['harbin', 'porto', 'india']
CITIES = ['india']
# 假设数据文件的路径结构为：'data/[city]/[city]_edge.txt' 和 'data/[city]/[city]_node.txt'
# 请根据实际文件路径进行调整！
DATA_DIR_TEMPLATE = '/home/lizhuoran/distance/data/{city}/'
LINK_FILE_TEMPLATE = 'edges.txt' 
NODE_FILE_TEMPLATE = 'nodes.txt'
# 原始的 chengdu 文件名作为参考
# LINK_FILE_TEMPLATE = 'chengdu_link-mod.txt' 
# NODE_FILE_TEMPLATE = 'chengdu_node-mod.txt' 

# --- 主处理函数 ---

def process_city_graph(city_name):
    """
    读取指定城市的节点和边数据，构建有向图，执行权重过滤，提取最大强连通分量，
    并计算最短距离矩阵。
    """
    print(f"\n--- 🚀 开始处理城市: {city_name.upper()} ---")
    
    # 构造文件路径
    data_path = DATA_DIR_TEMPLATE.format(city=city_name)
    link_file = os.path.join(data_path, LINK_FILE_TEMPLATE.format(city=city_name))
    node_file = os.path.join(data_path, NODE_FILE_TEMPLATE.format(city=city_name))
    
    # 检查文件是否存在
    if not os.path.exists(link_file) or not os.path.exists(node_file):
        print(f"⚠️ 错误: 找不到城市 {city_name} 的文件。请检查路径：")
        print(f"  边文件: {link_file}")
        print(f"  节点文件: {node_file}")
        return

    # 1. 读取边和节点数据
    try:
        # 注意: 假设 links 文件包含 'Node_Start', 'Node_End', 'Length' 三列
        links = pd.read_csv(link_file, header=0, sep=',')
        nodes = pd.read_csv(node_file, header=0, sep=',')
        print(f"✅ 读取数据成功。节点数: {len(nodes)}, 边数: {len(links)}")
    except Exception as e:
        print(f"❌ 读取 CSV 文件时发生错误: {e}")
        return

    # 2. 构建有向图
    start_col, end_col, weight_col = 'Node_Start', 'Node_End', 'Length'
    G = nx.DiGraph()
    # 记录总尝试添加的边数
    total_links = len(links)
    
    # 过滤无效权重边并构建图
    valid_links = []
    
    for _, row in tqdm(links.iterrows(), total=total_links, desc="构建图并过滤"):
        try:
            # 确保权重可以转换为浮点数
            weight = float(row[weight_col])
            
            # 过滤：权重必须是正数且是有限的 (Dijkstra和Node2Vec要求)
            if np.isfinite(weight) and weight > 0:
                # 存入有效链接列表
                valid_links.append((row[start_col], row[end_col], weight))
            else:
                pass # 忽略无效/非正/非有限权重
                
        except (ValueError, TypeError):
            pass # 忽略无法转换为数字的权重

    # 批量添加有效边
    G.add_weighted_edges_from(valid_links)
    
    # 报告过滤结果
    filtered_links_count = total_links - len(valid_links)
    if filtered_links_count > 0:
        print(f"⚠️ 过滤掉了 {filtered_links_count} 条权重为非正、非有限或格式错误的边。")

    print(f"✅ 初始图构建完成。节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 3. 检查并提取最大强连通分量 (SCC)
    if not nx.is_strongly_connected(G):
        print('⚠️ 图不是强连通的，正在提取最大强连通分量...')
        scs = list(nx.strongly_connected_components(G))
        # 找到最大的 SCC
        max_sc = max(scs, key=len)
        # 提取子图
        G = G.subgraph(max_sc).copy()
        print(f'✅ 最大 SCC 提取完成。新节点数: {G.number_of_nodes()}, 新边数: {G.number_of_edges()}')
    else:
        print('✅ 图已经是强连通的。')

    # 4. 保存图
    output_graph_file = os.path.join(data_path, f'{city_name}_graph_sc.pkl')
    with open(output_graph_file, 'wb') as f:
        pickle.dump(G, f)
        f.close()
    print(f"✅ 强连通图已保存到: {output_graph_file}")

    # 5. 计算最短距离矩阵
    n = G.number_of_nodes()
    
    if n == 0:
        print("❌ 警告: 强连通图为空，跳过距离矩阵计算。")
        return # 如果图为空，则直接返回

    # 将节点重新映射到连续的整数索引 [0, n-1]
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    index_to_node = {i: node for node, i in node_to_index.items()}
    
    distance_matrix = np.full((n, n), np.inf) # 初始化为无穷大
    
    print(f"⏳ 开始计算最短距离矩阵 ({n}x{n})...")

    # 遍历所有节点计算单源最短路径
    # 使用 tqdm 显示进度
    for s_index, s in tqdm(index_to_node.items(), total=n, desc="计算距离"):
        # 使用 Dijkstra 算法计算从源节点 s 到所有其他节点的最短路径长度
        # 注意: 权重名称必须是 'weight'
        length = nx.single_source_dijkstra_path_length(G, s, weight='weight') 
        for t, dist in length.items():
            t_index = node_to_index[t]
            distance_matrix[s_index, t_index] = dist

    # 6. 保存距离矩阵
    output_matrix_file = os.path.join(data_path, f'{city_name}_directed_shortest_distance_matrix.npy')
    np.save(output_matrix_file, distance_matrix)
    print(f"✅ 最短距离矩阵已保存到: {output_matrix_file}")
    


# --- 脚本执行入口 ---

if __name__ == '__main__':
    # 遍历所有城市并处理
    for city in CITIES:
        process_city_graph(city)

    print("\n--- ✅ 所有城市处理完毕 (end) ---")
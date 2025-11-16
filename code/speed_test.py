import time
import numpy as np
import pickle
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra

from distnet_model import ImprovedMultiLayerPerceptron
from config import get_config

# 加载配置
config, _ = get_config()

# 加载最短距离矩阵
sdm = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
maxLength = np.max(sdm)
sdm = sdm / maxLength

# 加载node2vec嵌入
with open("param/dist2vec_embed.pkl", 'rb') as f:
    embed = pickle.load(f)
embed = np.array(list(embed.values()))
embed = (embed - embed.min()) / (embed.max() - embed.min())

# 加载节点坐标
node_long_lat = pd.read_csv("data/chengdu_node-mod.txt", header=0, sep=',')
node_long_lat = np.array(node_long_lat)[:, 1:3]
node_long_lat[:, 0] = (node_long_lat[:, 0] - node_long_lat[:, 0].min()) / (node_long_lat[:, 0].max() - node_long_lat[:, 0].min())
node_long_lat[:, 1] = (node_long_lat[:, 1] - node_long_lat[:, 1].min()) / (node_long_lat[:, 1].max() - node_long_lat[:, 1].min())

def estimate_distance(model, embed, node_long_lat, i, j, maxLength):
    # 用distnet_model进行推理
    x1 = np.concatenate((embed[i], node_long_lat[i]), axis=0)
    x2 = np.concatenate((embed[j], node_long_lat[j]), axis=0)
    x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
    x2_tensor = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x1_tensor, x2_tensor)
        dist = output.item() * maxLength  # 反归一化
    return dist

def test_speed(num_tests=100, batch_mode=False, batch_size=32):
    """
    用distnet_model进行距离推理速度测试
    支持串行和批量（并行）推理
    """
    # 初始化模型
    embed_dim = config.embed_dim
    long_lat_embed_dim = config.long_lat_embed_dim
    input_dim = embed_dim + long_lat_embed_dim
    hidden_dim1 = 512
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = config.n_output
    device = torch.device('cpu')
    model = ImprovedMultiLayerPerceptron(input_dim * 2, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

    # 加载权重
    if config.type == 1:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_1.ckpt", map_location=device))
    elif config.type == 2:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_tilde_L1.ckpt", map_location=device))
    elif config.type == 3:
        model.load_state_dict(torch.load("param/distnet_best_chengdu_L1.ckpt", map_location=device))
    model.eval()

    n_nodes = sdm.shape[0]
    results = []
    if not batch_mode:
        # 串行推理
        total_time = 0
        for _ in range(num_tests):
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            t0 = time.time()
            dist = estimate_distance(model, embed, node_long_lat, i, j, maxLength)
            t1 = time.time()
            total_time += (t1 - t0)
            results.append(dist)
        avg_time = total_time / num_tests
        print(f"串行测试次数: {num_tests}")
        print(f"平均推理时间: {avg_time:.6f} 秒")
        print(f"距离估计样例: {results[:5]}")
    else:
        # 批量推理
        batch_num = num_tests // batch_size
        total_time = 0
        for _ in range(batch_num):
            idx_i = np.random.randint(0, n_nodes, batch_size)
            idx_j = np.random.randint(0, n_nodes, batch_size)
            x1 = np.concatenate([np.concatenate((embed[i], node_long_lat[i]), axis=0)[None, :] for i in idx_i], axis=0)
            x2 = np.concatenate([np.concatenate((embed[j], node_long_lat[j]), axis=0)[None, :] for j in idx_j], axis=0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32)
            x2_tensor = torch.tensor(x2, dtype=torch.float32)
            t0 = time.time()
            with torch.no_grad():
                output = model(x1_tensor, x2_tensor)
                dists = output.squeeze().cpu().numpy() * maxLength
            t1 = time.time()
            total_time += (t1 - t0)
            results.extend(dists.tolist())
        avg_time = total_time / num_tests
        print(f"批量测试次数: {num_tests}（每批{batch_size}）")
        print(f"平均推理时间: {avg_time:.6f} 秒")
        print(f"距离估计样例: {results[:5]}")

def test_dijkstra_speed(num_tests=100):
    """
    测试Dijkstra算法串行推理速度
    """
    # 构造邻接矩阵（0表示无边，非零表示距离）
    adj_matrix = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
    n_nodes = adj_matrix.shape[0]
    results = []
    total_time = 0
    for _ in range(num_tests):
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        t0 = time.time()
        # Dijkstra单源最短路
        dist = dijkstra(adj_matrix, directed=True, indices=i)[j]
        t1 = time.time()
        total_time += (t1 - t0)
        results.append(dist)
    avg_time = total_time / num_tests
    print(f"Dijkstra串行测试次数: {num_tests}")
    print(f"Dijkstra平均推理时间: {avg_time:.6f} 秒")
    print(f"Dijkstra距离样例: {results[:5]}")

if __name__ == "__main__":
    # 串行推理
    test_speed(num_tests=100, batch_mode=False)
    # 批量推理
    test_speed(num_tests=100, batch_mode=True, batch_size=32)
    test_dijkstra_speed(num_tests=100)
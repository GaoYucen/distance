#!/usr/bin/env python3
"""
VLDB Distance — 一键数据准备脚本
===============================
从 OSMnx 下载 → .nodes/.edges → 生成 query pairs 

用法:
    python prepare_data.py                          # 全量，断点续传
    python prepare_data.py --city Beijing           # 单个城市
    python prepare_data.py --force                  # 强制重新下载
    python prepare_data.py --num_queries 200000     # 自定义 query 数量

数据量 (per city, 45 train pairs/node):
    - Harbin    (~43K 节点)   → ~1.9M pairs
    - Chengdu   (~111K 节点)  → ~5.0M pairs
    - Qingdao   (~119K 节点)  → ~5.4M pairs
    - Beijing   (~163K 节点)  → ~7.3M pairs

required: pip install osmnx networkx routingkit_cch tqdm
"""

import argparse
import os
import sys
import time
import pickle
import random
import gc
import numpy as np
import networkx as nx

# ============================================================
# 配置
# ============================================================

CITIES = {
    "Harbin": {
        "query": "Harbin, Heilongjiang, China",
        "network_type": "drive",
    },
    "Chengdu": {
        "query": "Chengdu, Sichuan, China",
        "network_type": "drive",
    },
    "Qingdao": {
        "query": "Qingdao, Shandong, China",
        "network_type": "drive",
    },
    "Beijing": {
        "query": "Beijing, China",
        "network_type": "drive",
    },
}

# 各城市的 query 策略: ("proportional" 或 ("fixed", 数量))
# 这个proportional是根据之前的综述研究中每个城市查询的数量与之节点的比例关系确定的
CITY_QUERY_STRATEGY = {
    "Harbin":       ("proportional", 45),
    "Chengdu":      ("proportional", 45),
    "Qingdao":      ("proportional", 45),
    "Beijing":      ("proportional", 45),
}

TRAIN_RATIO = 0.8
SEED = 42
OSMNX_TIMEOUT = 180  # 秒


# ============================================================
# Stage 1: OSMnx 下载
# ============================================================

def download_osmnx(city_name, query, network_type, data_dir, force=False):
    """
    从 OSMnx 下载路网, 保存为 pickle。支持断点续传。
    """
    pkl_path = os.path.join(data_dir, f"{city_name}_osmnx.pkl")

    if os.path.exists(pkl_path) and not force:
        try:
            G = pickle.load(open(pkl_path, 'rb'))
            n, e = G.number_of_nodes(), G.number_of_edges()
            print(f"  [跳过] {city_name}: 已存在 ({n} 节点, {e} 边)")
            return G
        except Exception:
            print(f"  [警告] pickle 损坏, 重新下载 {city_name}")

    print(f"  [下载] {city_name} <- '{query}' ...")
    t0 = time.time()

    try:
        import osmnx as ox
        ox.settings.timeout = OSMNX_TIMEOUT
        G = ox.graph_from_place(query, network_type=network_type)
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None

    os.makedirs(data_dir, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f)

    n, e = G.number_of_nodes(), G.number_of_edges()
    elapsed = time.time() - t0
    mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  ✓ {city_name}: {n} 节点, {e} 边, {elapsed:.0f}s, {mb:.1f}MB")
    return G


# ============================================================
# Stage 2: OSMnx → .nodes / .edges
# ============================================================

def export_to_nodes_edges(city_name, G, data_dir, force=False):
    """
    将 OSMnx MultiDiGraph 转换为 .nodes / .edges 格式。

    .nodes:  idx,lon,lat
    .edges:  u,v,length

    断点续传: 如果已存在则跳过。
    """
    city_dir = os.path.join(data_dir, f"OSM_{city_name}")
    nodes_path = os.path.join(city_dir, f"OSM_{city_name}.nodes")
    edges_path = os.path.join(city_dir, f"OSM_{city_name}.edges")

    if os.path.exists(nodes_path) and os.path.exists(edges_path) and not force:
        print(f"  [跳过] OSM_{city_name}: .nodes/.edges 已存在")
        return

    if G is None:
        pkl_path = os.path.join(data_dir, f"{city_name}_osmnx.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"未找到 pickle: {pkl_path}")
        G = pickle.load(open(pkl_path, 'rb'))

    os.makedirs(city_dir, exist_ok=True)
    print(f"  [转换] {city_name} → .nodes / .edges ...")

    # ---- 节点映射 ----
    original_nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(original_nodes)}

    with open(nodes_path, 'w') as f:
        for osm_id in original_nodes:
            idx = node_to_idx[osm_id]
            lon = G.nodes[osm_id]['x']
            lat = G.nodes[osm_id]['y']
            f.write(f"{idx},{lon},{lat}\n")

    # ---- 边处理: 去重保留最短 ----
    edge_dict = {}
    for u_osm, v_osm, k, data in G.edges(keys=True, data=True):
        u = node_to_idx[u_osm]
        v = node_to_idx[v_osm]
        length = data.get('length', 0)
        key = (u, v)
        if key not in edge_dict or length < edge_dict[key]:
            edge_dict[key] = length

    with open(edges_path, 'w') as f:
        for (u, v), length in sorted(edge_dict.items()):
            f.write(f"{u},{v},{length:.6f}\n")

    n_nodes = len(original_nodes)
    n_edges = len(edge_dict)
    oneway_count = sum(1 for _, _, _, d in G.edges(keys=True, data=True)
                       if d.get('oneway', False))
    print(f"  ✓ OSM_{city_name}: {n_nodes} 节点, {n_edges} 边 "
          f"(oneway {oneway_count}, {100*oneway_count/max(1,n_edges):.1f}%)")


# ============================================================
# Stage 3: 生成 query pairs (CCH 精确最短路径)
# ============================================================

def load_directed_graph(data_dir, city_name):
    """
    从 .nodes/.edges 加载 nx.DiGraph, 返回 CCH 所需的数组。

    Returns:
        G, node_count, tail, head, weights(x10), lat, lon
    """
    city_dir = os.path.join(data_dir, f"OSM_{city_name}")
    nodes_file = os.path.join(city_dir, f"OSM_{city_name}.nodes")
    edges_file = os.path.join(city_dir, f"OSM_{city_name}.edges")

    G = nx.DiGraph()
    tail, head, weights = [], [], []
    max_nid = 0

    with open(edges_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2])
            G.add_edge(u, v, weight=w)
            tail.append(u); head.append(v)
            weights.append(int(w * 10))     # 0.1m 精度
            max_nid = max(max_nid, u, v)

    node_count = max_nid + 1
    lat = [0.0] * node_count
    lon = [0.0] * node_count

    with open(nodes_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            nid = int(parts[0])
            lat[nid] = float(parts[2])
            lon[nid] = float(parts[1])

    return G, node_count, tail, head, weights, lat, lon


def build_cch(node_count, tail, head, weights, lat, lon):
    """
    用 inertial flow ordering 构建 CCH 索引。
    """
    import routingkit_cch as rk

    print("    [1/3] inertial flow 节点排序 (空间剖分)...")
    t0 = time.time()
    order = rk.compute_order_inertial(node_count, tail, head, lat, lon)
    print(f"          耗时 {time.time() - t0:.1f}s")

    print("    [2/3] 构建 CCH 图 (图收缩)...")
    t1 = time.time()
    cch = rk.CCH(order, tail, head, False)
    print(f"          耗时 {time.time() - t1:.1f}s")

    print("    [3/3] 构建 Metric (边权重绑定)...")
    t2 = time.time()
    metric = rk.CCHMetric(cch, weights)
    print(f"          耗时 {time.time() - t2:.1f}s")

    return rk.CCHQuery(metric)


def generate_queries(G, query_obj, node_count, num_queries, seed):
    """
    随机采样 OD 对, CCH 批量查询双向距离。
    """
    random.seed(seed)
    nodes = list(range(node_count))

    queries = []
    filled = 0
    max_attempts = num_queries * 5

    def _dist(src, dst):
        try:
            r = query_obj.run(int(src), int(dst))
            d = r.distance / 10.0
            del r
            return d
        except Exception:
            return -1.0

    from tqdm import tqdm
    pbar = tqdm(total=num_queries, desc=f"    Query", unit="对")

    batch_size = 2000
    while filled < num_queries and filled * 3 < max_attempts:
        need = min(batch_size, (num_queries - filled) * 3)
        src_batch = random.choices(nodes, k=need)
        dst_batch = random.choices(nodes, k=need)

        for u, v in zip(src_batch, dst_batch):
            if u == v:
                continue
            d_uv = _dist(u, v)
            if d_uv < 0:
                continue
            d_vu = _dist(v, u)
            queries.append([u, v, d_uv, d_vu])
            filled += 1
            pbar.update(1)
            if filled >= num_queries:
                break

    pbar.close()
    ratio = filled / max(1, filled * 3) if filled > 0 else 0
    print(f"    完成: {filled:,} 对 (有效比 ~{ratio:.1%})")
    return queries


def save_queries(queries, output_dir, city_name, train_ratio=0.8, seed=42):
    """80/20 train/test split, 保存为 .queries 格式。"""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    shuffled = queries[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)

    train = shuffled[:split]
    test = shuffled[split:]

    train_path = os.path.join(output_dir, f"{city_name}_train.queries")
    with open(train_path, 'w') as f:
        f.write(f"# {city_name} train queries (n={len(train)})\n")
        f.write("# u,v,d_uv,d_vu\n")
        for u, v, d_uv, d_vu in train:
            f.write(f"{u},{v},{d_uv:.1f},{d_vu:.1f}\n")

    test_path = os.path.join(output_dir, f"{city_name}_test.queries")
    with open(test_path, 'w') as f:
        f.write(f"# {city_name} test queries (n={len(test)})\n")
        f.write("# u,v,d_uv,d_vu\n")
        for u, v, d_uv, d_vu in test:
            f.write(f"{u},{v},{d_uv:.1f},{d_vu:.1f}\n")

    print(f"    保存: train={len(train):,} → {train_path}")
    print(f"          test ={len(test):,} → {test_path}")
    return train_path, test_path


def process_queries(city_name, data_dir, force=False):
    """
    为单个城市生成 query files (CCH 双向距离)。
    断点续传: 如果 target dir 已有 train+test .queries 则跳过。
    """
    city_dir = os.path.join(data_dir, f"OSM_{city_name}")

    strategy, param = CITY_QUERY_STRATEGY[city_name]
    if strategy == "fixed":
        num_queries = param
        query_subdir = "random_500k"
    else:
        # proportional: load graph to get node count
        G, n_nodes, _, _, _, _, _ = load_directed_graph(data_dir, city_name)
        num_queries = n_nodes * param
        query_subdir = "proportional"

    output_dir = os.path.join(city_dir, query_subdir)
    train_path = os.path.join(output_dir, f"{city_name}_train.queries")
    test_path = os.path.join(output_dir, f"{city_name}_test.queries")

    if os.path.exists(train_path) and os.path.exists(test_path) and not force:
        print(f"  [跳过] {city_name}/{query_subdir}: queries 已存在 "
              f"({num_queries:,} pairs target)")
        # Quick validation: at least 1 line
        with open(train_path) as f:
            count = sum(1 for l in f if not l.startswith('#'))
        print(f"         train={count:,} 对")
        return

    print(f"  [Query] {city_name}: target={num_queries:,} pairs "
          f"({'random' if strategy == 'fixed' else '45/node'})")

    t0 = time.time()

    G, node_count, tail, head, weights, lat, lon = load_directed_graph(data_dir, city_name)

    query_obj = build_cch(node_count, tail, head, weights, lat, lon)

    queries = generate_queries(G, query_obj, node_count, num_queries, SEED)

    del query_obj; gc.collect()

    save_queries(queries, output_dir, city_name, train_ratio=TRAIN_RATIO, seed=SEED)

    # 统计
    d_uv = [q[2] for q in queries if q[2] > 0]
    asym = sum(1 for q in queries
               if q[2] > 0 and q[3] > 0 and abs(q[2] - q[3]) > 1.0)
    print(f"    非对称 pair: {asym}/{len(queries)} "
          f"({100*asym/max(1,len(queries)):.1f}%)")
    print(f"    耗时: {time.time() - t0:.0f}s")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLDB Distance 数据准备: OSMnx → .nodes/.edges → queries")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data"),
                        help="数据目录 (默认: ../data)")
    parser.add_argument("--city", type=str, default=None,
                        help="只处理指定城市 (如 Beijing)")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="覆盖固定 query 数量 (仅对 random_500k 模式有效)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载/处理所有阶段")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过 OSMnx 下载 (需要已存在 pickle)")
    parser.add_argument("--skip-queries", action="store_true",
                        help="跳过 query 生成")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    targets = CITIES
    if args.city:
        if args.city not in CITIES:
            print(f"未知城市 '{args.city}'，可选: {list(CITIES.keys())}")
            sys.exit(1)
        targets = {args.city: CITIES[args.city]}

    if args.num_queries:
        for k in CITY_QUERY_STRATEGY:
            if CITY_QUERY_STRATEGY[k][0] == "fixed":
                CITY_QUERY_STRATEGY[k] = ("fixed", args.num_queries)

    print(f"{'=' * 60}")
    print(f"VLDB Distance — 数据准备")
    print(f"{'=' * 60}")
    print(f"城市: {list(targets.keys())}")
    print(f"数据目录: {data_dir}")
    print(f"模式: {'强制' if args.force else '断点续传'}")
    print()

    for city_name, cfg in targets.items():
        print(f"{'─' * 50}")
        print(f"[{city_name}]")
        print(f"{'─' * 50}")

        # ---- Stage 1: Download ----
        if not args.skip_download:
            G = download_osmnx(city_name, cfg["query"], cfg["network_type"],
                               data_dir, force=args.force)
            if G is None:
                continue

        # ---- Stage 2: Convert ----
        export_to_nodes_edges(city_name, None, data_dir, force=args.force)

        # ---- Stage 3: Queries ----
        if not args.skip_queries:
            process_queries(city_name, data_dir, force=args.force)

        print()

    print("=" * 60)
    print("全部完成!")
    print("=" * 60)

    # 数据大小汇总
    total_mb = 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            total_mb += os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
    print(f"数据总量: {total_mb:.1f} MB")


if __name__ == '__main__':
    main()

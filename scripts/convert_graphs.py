#!/usr/bin/env python3
"""Convert node/edge xlsx files in dataset folders to txt + config.json.

Usage:
    python scripts/convert_graphs.py --data-dir /home/lizhuoran/distance/data

Behavior/assumptions:
- For each immediate subdirectory under --data-dir, the script looks for a node file
  (named containing 'node' or 'nodes' and ending with .xlsx/.xls) and an edge file
  (containing 'edge' or 'edges').
- Node file must contain columns longitude, latitude, osmid (case-insensitive).
  Nodes are re-numbered from 0 upwards sorted by osmid (deterministic).
- Edge file must contain columns u, v, length (case-insensitive). u/v are osmids
  and will be remapped to new node ids; edges referencing unknown osmids are dropped.
- Output files written in the same folder:
    nodes.txt -> header `Node,Longitude,Latitude`
    edges.txt -> header `Node_Start,Node_End,Length`
    config.json -> contains stats including `directed_only_ratio`, `n_nodes`, `n_edges`.

"""

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_file(folder: Path, keywords):
    candidates = []
    for p in folder.iterdir():
        if p.is_file():
            name = p.name.lower()
            if any(k in name for k in keywords) and (name.endswith('.xlsx') or name.endswith('.xls') or name.endswith('.csv') or name.endswith('.txt')):
                # Avoid picking up the script's own output as input if other options exist
                if p.name in ['nodes.txt', 'edges.txt', 'nodes_lscc.txt', 'edges_lscc.txt']:
                    continue
                candidates.append(p)
    
    if not candidates:
        # Fallback to standard names if no other candidates found
        for name in ['nodes.txt', 'edges.txt']:
            if any(k in name for k in keywords) and (folder / name).exists():
                return folder / name
        return None

    # Prioritize Excel/CSV over TXT for raw data discovery
    for p in candidates:
        if p.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            return p
    return candidates[0]


def read_nodes(node_path: Path):
    logging.info("Reading node file: %s", node_path)
    # Support xlsx/xls and csv
    if node_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(node_path, engine='openpyxl')
    else:
        df = pd.read_csv(node_path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    
    # 增加更灵活的列名映射
    lon_col = next((cols[c] for c in ['longitude', 'lng', 'lon', 'x'] if c in cols), None)
    lat_col = next((cols[c] for c in ['latitude', 'lat', 'y'] if c in cols), None)
    id_col = next((cols[c] for c in ['osmid', 'id', 'node_id', 'node'] if c in cols), None)

    if not all([lon_col, lat_col, id_col]):
        raise ValueError(f"Node file {node_path} missing required columns (found: {list(df.columns)})")
    
    df = df.rename(columns={lon_col: 'longitude', lat_col: 'latitude', id_col: 'osmid'})
    # Drop rows with missing osmid
    df = df.dropna(subset=['osmid'])
    # Ensure osmid is int-like
    df['osmid'] = df['osmid'].astype(int)
    return df


def read_edges(edge_path: Path):
    logging.info("Reading edge file: %s", edge_path)
    if edge_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(edge_path, engine='openpyxl')
    else:
        df = pd.read_csv(edge_path)
    cols = {c.lower(): c for c in df.columns}
    
    u_col = next((cols[c] for c in ['u', 'source', 'from', 'node_start', 'start_node'] if c in cols), None)
    v_col = next((cols[c] for c in ['v', 'target', 'to', 'node_end', 'end_node'] if c in cols), None)
    w_col = next((cols[c] for c in ['length', 'dist', 'weight', 'w'] if c in cols), None)

    if not all([u_col, v_col, w_col]):
        raise ValueError(f"Edge file {edge_path} missing required columns (found: {list(df.columns)})")

    df = df.rename(columns={u_col: 'u', v_col: 'v', w_col: 'length'})
    # Drop rows missing u or v
    df = df.dropna(subset=['u', 'v'])
    # Cast to int for osmids when possible
    df['u'] = df['u'].astype(int)
    df['v'] = df['v'].astype(int)
    # Filter out non-positive lengths (Dijkstra requires positive weights)
    if 'length' in df.columns:
        df = df[df['length'] > 0]
    return df


def convert_folder(folder: Path, sort_by='osmid'):
    logging.info('\nProcessing folder: %s', folder)
    node_file = find_file(folder, ['node', 'nodes'])
    edge_file = find_file(folder, ['edge', 'edges'])
    if node_file is None:
        logging.warning('No node file found in %s; skipping', folder)
        return
    if edge_file is None:
        logging.warning('No edge file found in %s; continuing with nodes only', folder)

    df_nodes = read_nodes(node_file)
    
    # Determine ordering: sort by osmid by default
    if sort_by == 'osmid':
        df_nodes = df_nodes.sort_values('osmid').reset_index(drop=True)
    else:
        df_nodes = df_nodes.reset_index(drop=True)
    # create mapping osmid -> new id
    df_nodes['new_id'] = range(len(df_nodes))
    osmid_to_new = dict(zip(df_nodes['osmid'].tolist(), df_nodes['new_id'].tolist()))

    # write nodes txt
    nodes_out = folder / 'nodes.txt'
    with nodes_out.open('w', encoding='utf-8') as f:
        f.write('Node,Longitude,Latitude\n')
        for _, row in df_nodes.iterrows():
            f.write(f"{int(row['new_id'])},{float(row['longitude'])},{float(row['latitude'])}\n")
    logging.info('Wrote nodes to %s (%d nodes)', nodes_out, len(df_nodes))

    n_edges_written = 0
    directed_only_ratio = None

    # 初始化 config 字典，避免在 LSCC 处理时出现 UnboundLocalError
    cfg = {
        'n_nodes': len(df_nodes),
        'n_edges': 0,
        'directed_only_ratio': None,
    }

    if edge_file is not None:
        df_edges = read_edges(edge_file)
        # map u/v osmids to new ids, drop unknowns
        def map_osmid(x):
            return osmid_to_new.get(int(x), None)
        df_edges['u_new'] = df_edges['u'].map(map_osmid)
        df_edges['v_new'] = df_edges['v'].map(map_osmid)
        before = len(df_edges)
        if before > 0 and df_edges['u_new'].isnull().all():
            logging.warning("!!! ALL edges dropped. This usually means the IDs in the edge file do not match the IDs in the node file.")
            
        df_edges = df_edges.dropna(subset=['u_new', 'v_new'])
        df_edges['u_new'] = df_edges['u_new'].astype(int)
        df_edges['v_new'] = df_edges['v_new'].astype(int)
        after = len(df_edges)
        logging.info('Edges before filter: %d, after dropping missing nodes: %d', before, after)

        # write edges out
        edges_out = folder / 'edges.txt'
        with edges_out.open('w', encoding='utf-8') as f:
            f.write('Node_Start,Node_End,Length\n')
            for _, row in df_edges.iterrows():
                f.write(f"{int(row['u_new'])},{int(row['v_new'])},{float(row['length'])}\n")
                n_edges_written += 1
        logging.info('Wrote edges to %s (%d edges)', edges_out, n_edges_written)

        # 更新 config 中的边总数
        cfg['n_edges'] = n_edges_written

        # compute directed-only ratio: fraction of edges that don't have the reverse
        edge_pairs = set((int(r.u_new), int(r.v_new)) for r in df_edges.itertuples())
        n_directed_only = sum(1 for (a, b) in edge_pairs if (b, a) not in edge_pairs)
        total = len(edge_pairs)
        directed_only_ratio = float(n_directed_only) / float(total) if total > 0 else 0.0
        logging.info('Directed-only edges: %d / %d = %.4f', n_directed_only, total, directed_only_ratio)
        cfg['directed_only_ratio'] = directed_only_ratio

        # --- 新增：生成强连通分量 (LSCC) 专用文件 ---
        # 使用原始 osmid 构建图来提取 LSCC
        logging.info("Checking connectivity for LSCC generation...")
        G = nx.DiGraph()
        edges_for_G = list(zip(df_edges['u'], df_edges['v']))
        G.add_edges_from(edges_for_G)
        
        if G.number_of_nodes() > 0 and not nx.is_strongly_connected(G):
            logging.info("Graph is NOT strongly connected. Generating *_lscc.txt files...")
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            logging.info(f"LSCC nodes: {len(largest_cc)} (Original: {G.number_of_nodes()})")
            
            # 1. 提取 LSCC 节点并重编号
            df_nodes_lscc = df_nodes[df_nodes['osmid'].isin(largest_cc)].copy()
            if sort_by == 'osmid':
                df_nodes_lscc = df_nodes_lscc.sort_values('osmid').reset_index(drop=True)
            else:
                df_nodes_lscc = df_nodes_lscc.reset_index(drop=True)
            
            df_nodes_lscc['new_id'] = range(len(df_nodes_lscc))
            lscc_osmid_to_new = dict(zip(df_nodes_lscc['osmid'].tolist(), df_nodes_lscc['new_id'].tolist()))
            
            # 2. 写入 nodes_lscc.txt
            nodes_lscc_out = folder / 'nodes_lscc.txt'
            with nodes_lscc_out.open('w', encoding='utf-8') as f:
                f.write('Node,Longitude,Latitude\n')
                for _, row in df_nodes_lscc.iterrows():
                    f.write(f"{int(row['new_id'])},{float(row['longitude'])},{float(row['latitude'])}\n")
            
            # 3. 提取 LSCC 边并重映射
            df_edges_lscc = df_edges[df_edges['u'].isin(largest_cc) & df_edges['v'].isin(largest_cc)].copy()
            df_edges_lscc['u_new'] = df_edges_lscc['u'].map(lscc_osmid_to_new)
            df_edges_lscc['v_new'] = df_edges_lscc['v'].map(lscc_osmid_to_new)
            df_edges_lscc = df_edges_lscc.dropna(subset=['u_new', 'v_new']) # Should be empty drop but safe to keep
            
            # 4. 写入 edges_lscc.txt
            edges_lscc_out = folder / 'edges_lscc.txt'
            with edges_lscc_out.open('w', encoding='utf-8') as f:
                f.write('Node_Start,Node_End,Length\n')
                for _, row in df_edges_lscc.iterrows():
                    f.write(f"{int(row['u_new'])},{int(row['v_new'])},{float(row['length'])}\n")
            
            logging.info(f"Wrote LSCC files: {nodes_lscc_out.name}, {edges_lscc_out.name}")
            
            # 更新 config 以包含 LSCC 信息
            cfg['n_nodes_lscc'] = len(df_nodes_lscc)
            cfg['n_edges_lscc'] = len(df_edges_lscc)
        elif G.number_of_nodes() > 0:
            logging.info("Graph is already strongly connected. No separate LSCC files needed.")
        else:
            logging.warning("Graph has no edges; skipping LSCC check.")

    cfg_path = folder / 'config.json'
    with cfg_path.open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    logging.info('Wrote config to %s', cfg_path)


def main():
    parser = argparse.ArgumentParser(description='Convert node/edge xlsx -> txt and produce config.json')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to parent data directory containing dataset folders')
    parser.add_argument('--folders', type=str, nargs='*', default=None, help='Optional list of subfolder names to process (default: all)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f'data directory {data_dir} does not exist')

    subfolders = [p for p in data_dir.iterdir() if p.is_dir()]
    if args.folders:
        subfolders = [data_dir / nm for nm in args.folders]

    for folder in subfolders:
        try:
            convert_folder(folder)
        except Exception as e:
            logging.exception('Failed to process %s: %s', folder, e)


if __name__ == '__main__':
    main()

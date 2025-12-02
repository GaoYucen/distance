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

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_file(folder: Path, keywords):
    for p in folder.iterdir():
        if p.is_file():
            name = p.name.lower()
            if any(k in name for k in keywords) and (name.endswith('.xlsx') or name.endswith('.xls') or name.endswith('.csv')):
                return p
    return None


def read_nodes(node_path: Path):
    logging.info("Reading node file: %s", node_path)
    # Support xlsx/xls and csv
    if node_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(node_path, engine='openpyxl')
    else:
        df = pd.read_csv(node_path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    required = ['longitude', 'latitude', 'osmid']
    for r in required:
        if r not in cols:
            raise ValueError(f"Node file {node_path} missing required column '{r}' (found: {list(df.columns)})")
    df = df.rename(columns={cols['longitude']: 'longitude', cols['latitude']: 'latitude', cols['osmid']: 'osmid'})
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
    required = ['u', 'v', 'length']
    for r in required:
        if r not in cols:
            raise ValueError(f"Edge file {edge_path} missing required column '{r}' (found: {list(df.columns)})")
    df = df.rename(columns={cols['u']: 'u', cols['v']: 'v', cols['length']: 'length'})
    # Drop rows missing u or v
    df = df.dropna(subset=['u', 'v'])
    # Cast to int for osmids when possible
    df['u'] = df['u'].astype(int)
    df['v'] = df['v'].astype(int)
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
    if edge_file is not None:
        df_edges = read_edges(edge_file)
        # map u/v osmids to new ids, drop unknowns
        def map_osmid(x):
            return osmid_to_new.get(int(x), None)
        df_edges['u_new'] = df_edges['u'].map(map_osmid)
        df_edges['v_new'] = df_edges['v'].map(map_osmid)
        before = len(df_edges)
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

        # compute directed-only ratio: fraction of edges that don't have the reverse
        edge_pairs = set((int(r.u_new), int(r.v_new)) for r in df_edges.itertuples())
        n_directed_only = sum(1 for (a, b) in edge_pairs if (b, a) not in edge_pairs)
        total = len(edge_pairs)
        directed_only_ratio = float(n_directed_only) / float(total) if total > 0 else 0.0
        logging.info('Directed-only edges: %d / %d = %.4f', n_directed_only, total, directed_only_ratio)

    # write config.json
    cfg = {
        'n_nodes': len(df_nodes),
        'n_edges': n_edges_written,
        'directed_only_ratio': directed_only_ratio,
    }
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

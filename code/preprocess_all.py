#!/usr/bin/env python3
"""
Generalized preprocessing for multiple city datasets under `data/`.

For each city (either a directory under `data/` or standalone files like
`chengdu_node-mod.txt` / `chengdu_link-mod.txt`), the script will:

- Read nodes file (multiple heuristics supported) and build a mapping
  original_id -> contiguous node id starting at 0.
- Read edges file and remap endpoints to contiguous ids; drop edges with
  missing endpoints.
- Build a directed graph and compute directed shortest-distance matrix (Dijkstra per node)
  unless a precomputed matrix exists.
- Attempt to load node embeddings if available (node2vec .emb or .pkl), else skip.
- Select landmarks using `farthest_selection` from `distnet_model` if available.
- Save outputs into `data/<city>/pre/`:
  - `preprocessed_sdm.npy` (n x n float32, np.inf for unreachable)
  - `preprocessed_embed.npy` (m x d) if embeddings found
  - `preprocessed_node_long_lat.npy` (n x 2 normalized lon/lat)
  - `preprocessed_node_long_lat_origin.npy` (n x 2 raw lon/lat)
  - `preprocessed_indices.npy` (K x 2 array of directed pairs with finite nonzero dist)
  - `preprocessed_LM_indices.npy` (L x 2 landmark pair indices)

Usage: run from repo root (so relative `data/` path resolves)
    python distance/code/preprocess_all.py --data-dir data

"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import math
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def find_node_file(data_dir: Path) -> Path | None:
    # look for common names
    # Priority: LSCC files first
    if (data_dir / "nodes_lscc.txt").exists():
        return data_dir / "nodes_lscc.txt"
    candidates = list(data_dir.glob("*node*.txt")) + list(data_dir.glob("nodes*.txt"))
    if candidates:
        return candidates[0]
    # fallback: look for <city>_node-*.txt at parent
    return None


def find_edges_file(data_dir: Path) -> Path | None:
    if (data_dir / "edges_lscc.txt").exists():
        return data_dir / "edges_lscc.txt"
    candidates = list(data_dir.glob("*edge*.txt")) + list(data_dir.glob("edges*.txt"))
    if candidates:
        return candidates[0]
    return None


def read_nodes(path: Path) -> tuple[np.ndarray, list]:
    """Return (coords ndarray n x 2 (lon,lat)), orig_ids list length n"""
    df = None
    # Try reading with header first
    for sep in [",", "\t", "\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"Cannot parse nodes file: {path}")

    original_cols = [str(c) for c in df.columns]
    cols = [c.lower() for c in original_cols]
    # mapping from lowercase -> original column name (keeps original for indexing)
    col_map = {original_cols[i].lower(): original_cols[i] for i in range(len(original_cols))}
    # heuristics: find lon/lat columns
    lon_col = None
    lat_col = None
    id_col = None
    for c in cols:
        if c in ("lon", "longitude", "x", "lng") and lon_col is None:
            lon_col = col_map[c]
        if c in ("lat", "latitude", "y") and lat_col is None:
            lat_col = col_map[c]
        if c in ("osmid", "id", "node_id", "nid") and id_col is None:
            id_col = col_map[c]

    if lon_col is None or lat_col is None:
        # try positional inference: find two float columns that look like lon/lat
        float_cols = []
        for c in original_cols:
            try:
                vals = pd.to_numeric(df[c], errors="coerce")
                num_finite = np.isfinite(vals).sum()
                if num_finite >= max(1, len(vals) // 2):
                    float_cols.append(c)
            except Exception:
                continue
        if len(float_cols) >= 2:
            # choose two columns where values in plausible lon/lat ranges
            chosen = None
            for i in range(len(float_cols)):
                for j in range(i + 1, len(float_cols)):
                    a = pd.to_numeric(df[float_cols[i]], errors="coerce").to_numpy(dtype=float)
                    b = pd.to_numeric(df[float_cols[j]], errors="coerce").to_numpy(dtype=float)
                    # check ranges
                    if (np.nanmin(a) > -180 and np.nanmax(a) < 180 and np.nanmin(b) > -90 and np.nanmax(b) < 90) or (
                        np.nanmin(b) > -180 and np.nanmax(b) < 180 and np.nanmin(a) > -90 and np.nanmax(a) < 90
                    ):
                        chosen = (float_cols[i], float_cols[j])
                        break
                if chosen:
                    break
            if chosen:
                lon_col, lat_col = chosen

    if id_col is None:
        # try first column as id if integer-like
        first = original_cols[0]
        try:
            if pd.api.types.is_integer_dtype(pd.to_numeric(df[first], errors="coerce")):
                id_col = first
        except Exception:
            id_col = None

    if lon_col is None or lat_col is None:
        raise RuntimeError(f"Could not detect lon/lat in nodes file: {path}")

    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    if id_col is not None:
        orig_ids = df[id_col].astype(object).tolist()
    else:
        # create synthetic ids from index
        orig_ids = [int(i) for i in range(len(df))]

    coords = np.vstack([lon, lat]).T
    return coords, orig_ids


def read_edges(path: Path) -> np.ndarray:
    # return array shape (m,3): u, v, length
    df = None
    for sep in [",", "\t", "\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", header=0)
            if df.shape[1] >= 2:
                break
        except Exception:
            df = None
    if df is None:
        # try no-header (columns will be numeric)
        df = pd.read_csv(path, sep=None, engine="python", header=None)

    # preserve original column names and build lowercase->original map
    original_cols = [str(c) for c in df.columns]
    cols = [c.lower() for c in original_cols]
    col_map = {original_cols[i].lower(): original_cols[i] for i in range(len(original_cols))}
    u_col = None
    v_col = None
    w_col = None
    for c in cols:
        if c in ("u", "source", "from", "s", "node_start", "node_startid") and u_col is None:
            u_col = col_map[c]
        if c in ("v", "target", "to", "t", "node_end", "node_endid") and v_col is None:
            v_col = col_map[c]
        if c in ("length", "dist", "weight", "w") and w_col is None:
            w_col = col_map[c]

    if u_col is None or v_col is None:
        # fallback to positional
        u_col = original_cols[0]
        v_col = original_cols[1]
    if w_col is None:
        if df.shape[1] >= 3:
            w_col = original_cols[2]
        else:
            # default weight 1.0
            df["__weight__"] = 1.0
            w_col = "__weight__"

    u = df[u_col].to_numpy()
    v = df[v_col].to_numpy()
    w = pd.to_numeric(df[w_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    out = np.vstack([u, v, w]).T
    return out


def try_load_embedding(data_dir: Path, orig_ids: list) -> np.ndarray | None:
    # try several filenames
    # 1) node2vec .emb (word2vec text) 'city_node2vec.emb'
    # 2) pickled dict 'dist2vec_embed.pkl' or 'node2vec.pkl'
    from pathlib import Path
    try:
        from gensim.models import KeyedVectors
    except Exception:
        KeyedVectors = None

    emb = None
    # search for *.emb
    emb_files = list(data_dir.glob("*.emb")) + list(data_dir.glob("*vec*"))
    if emb_files and KeyedVectors is not None:
        try:
            kv = KeyedVectors.load_word2vec_format(str(emb_files[0]), binary=False)
            # map orig_ids order to embeddings
            ev = []
            for oid in orig_ids:
                key = str(oid)
                if key in kv:
                    ev.append(kv[key])
                else:
                    ev.append(np.zeros(kv.vector_size, dtype=float))
            emb = np.vstack(ev)
            return emb
        except Exception:
            emb = None

    # try pickled dict
    pickles = list(data_dir.glob("*.pkl"))
    for p in pickles:
        try:
            import pickle

            with open(p, "rb") as f:
                obj = pickle.load(f)
            # if dict mapping id->vec
            if isinstance(obj, dict):
                ev = []
                # try keys as strings or ints
                for oid in orig_ids:
                    if oid in obj:
                        ev.append(np.asarray(obj[oid], dtype=float))
                    elif str(oid) in obj:
                        ev.append(np.asarray(obj[str(oid)], dtype=float))
                    else:
                        ev.append(np.zeros(len(next(iter(obj.values()))), dtype=float))
                emb = np.vstack(ev)
                return emb
        except Exception:
            continue

    return None


def _compute_single_source_from_edges(args):
    """Helper function for parallel computation - rebuilds graph from edges"""
    i, edges, n, directed = args
    try:
        # Rebuild graph for this worker (lightweight for single-source)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(n))
        edge_dict = {}
        for u, v, w in edges:
            uu, vv, ww = int(u), int(v), float(w)
            if 0 <= uu < n and 0 <= vv < n:
                key = (uu, vv)
                if key not in edge_dict or edge_dict[key] > ww:
                    edge_dict[key] = ww
        G.add_edges_from([(u, v, {"weight": w}) for (u, v), w in edge_dict.items()])
        
        # Compute shortest paths from node i
        lengths = nx.single_source_dijkstra_path_length(G, i, weight="weight")
        row = np.full(n, np.inf, dtype=float)
        for j, d in lengths.items():
            row[j] = d
        return i, row
    except Exception as e:
        logging.warning(f"Error computing paths from node {i}: {e}")
        return i, np.full(n, np.inf, dtype=float)


def build_sdm(n: int, edges: np.ndarray, directed: bool = True, 
              parallel: bool = True, n_jobs: int = None, 
              batch_size: int = None, use_sparse: bool = False) -> np.ndarray:
    """
    Build shortest distance matrix with optimizations for large graphs.
    
    Args:
        n: Number of nodes
        edges: Edge array with (u, v, weight)
        directed: Whether graph is directed
        parallel: Whether to use parallel computation
        n_jobs: Number of parallel workers (None = auto)
        batch_size: Process nodes in batches to save memory (None = process all)
        use_sparse: Use sparse matrix storage (saves memory but slower access)
    
    Returns:
        SDM matrix (n x n) or sparse matrix if use_sparse=True
    """
    logging.info(f"Building graph with {n} nodes and {len(edges)} edges...")
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    
    # Build graph more efficiently
    edge_dict = {}
    for u, v, w in edges:
        try:
            uu = int(u)
            vv = int(v)
            ww = float(w)
        except Exception:
            continue
        if uu < 0 or vv < 0 or uu >= n or vv >= n:
            continue
        # keep smallest weight if multiple edges
        key = (uu, vv)
        if key not in edge_dict or edge_dict[key] > ww:
            edge_dict[key] = ww
    
    G.add_edges_from([(u, v, {"weight": w}) for (u, v), w in edge_dict.items()])
    logging.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # For very large graphs, use sparse storage or batch processing
    if use_sparse:
        from scipy.sparse import dok_matrix
        # For sparse matrix, we don't initialize with inf - we only store finite values
        # Initialize as empty, then fill with inf where needed
        sdm = dok_matrix((n, n), dtype=float)
        # Note: sparse matrices don't have a fill method, we'll set values as we go
    else:
        sdm = np.full((n, n), np.inf, dtype=float)
    
    # Determine batch size based on available memory
    if batch_size is None:
        # Estimate memory: n * n * 8 bytes (float64) or n * n * 4 bytes (float32)
        # For large graphs, process in batches
        if n > 10000:
            batch_size = max(100, n // 100)  # Process 1% at a time
        else:
            batch_size = n
    
    # Parallel computation
    if parallel and n > 100:  # Only parallelize for larger graphs
        if n_jobs is None:
            n_jobs = min(cpu_count(), 8)  # Limit to 8 cores to avoid memory issues
        
        logging.info(f"Computing SDM in parallel with {n_jobs} workers...")
        
        # Convert edges to list for serialization (needed for multiprocessing)
        edges_list = [(int(u), int(v), float(w)) for u, v, w in edges 
                     if 0 <= int(u) < n and 0 <= int(v) < n]
        
        # Process in batches to manage memory
        node_indices = list(range(n))
        
        for batch_start in tqdm(range(0, n, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, n)
            batch_nodes = node_indices[batch_start:batch_end]
            
            # Prepare arguments for parallel processing (pass edges, not graph)
            args_list = [(i, edges_list, n, directed) for i in batch_nodes]
            
            # Use multiprocessing
            with Pool(processes=n_jobs) as pool:
                batch_results = pool.map(_compute_single_source_from_edges, args_list)
            
            # Store results
            for i, row in batch_results:
                if use_sparse:
                    # For sparse matrix, store all finite values (including 0)
                    # We'll handle inf conversion later
                    for j, val in enumerate(row):
                        if np.isfinite(val):
                            sdm[i, j] = val
                        # inf values are not stored (will be 0 in toarray(), we'll fix this)
                else:
                    sdm[i, :] = row
            
            # Clear intermediate results to free memory
            del batch_results
        
        if use_sparse:
            # Convert to CSR for efficient storage and access
            sdm = sdm.tocsr()
            logging.info(f"SDM computed (sparse, {sdm.nnz} non-zero entries out of {n*n} total)")
        else:
            logging.info(f"SDM computed (dense, {n}x{n})")
    else:
        # Sequential computation (for small graphs or when parallel=False)
        logging.info("Computing SDM sequentially...")
        for i in tqdm(range(n), desc="Computing shortest paths"):
            lengths = nx.single_source_dijkstra_path_length(G, i, weight="weight")
            if use_sparse:
                # Store all finite values in sparse matrix
                for j, d in lengths.items():
                    if np.isfinite(d):
                        sdm[i, j] = d
            else:
                for j, d in lengths.items():
                    sdm[i, j] = d
    
    return sdm


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    # coords: n x 2 lon,lat
    out = coords.astype(float).copy()
    for k in range(2):
        arr = out[:, k]
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx - mn > 0:
            out[:, k] = (arr - mn) / (mx - mn)
        else:
            out[:, k] = 0.0
    return out


def process_city(root: Path, city_name: str, data_dir: Path, 
                parallel: bool = True, n_jobs: int = None, 
                batch_size: int = None, use_sparse: bool = False):
    logging.info(f"Processing city {city_name} in {root}")
    out_pre = root / "pre"
    out_pre.mkdir(parents=True, exist_ok=True)

    # find node & edge files
    node_file = find_node_file(root)
    if node_file is None:
        # try top-level name
        candidates = [p for p in data_dir.glob(f"*{city_name}*node*.txt")]
        node_file = candidates[0] if candidates else None
    if node_file is None:
        logging.warning(f"No node file found for {city_name}, skipping")
        return

    edges_file = find_edges_file(root)
    if edges_file is None:
        candidates = [p for p in data_dir.glob(f"*{city_name}*edge*.txt")]
        edges_file = candidates[0] if candidates else None
    if edges_file is None:
        logging.warning(f"No edges file found for {city_name}, skipping")
        return

    coords, orig_ids = read_nodes(node_file)
    logging.info(f"Read {len(orig_ids)} nodes from {node_file}")

    # build mapping orig id -> new id
    id_to_new = {orig_ids[i]: i for i in range(len(orig_ids))}

    edges_raw = read_edges(edges_file)
    logging.info(f"Read {edges_raw.shape[0]} edges from {edges_file}")

    # remap edges to contiguous ids
    remapped = []
    for u, v, w in edges_raw:
        key_u = u
        key_v = v
        # try int or str matching
        found_u = None
        found_v = None
        if key_u in id_to_new:
            found_u = id_to_new[key_u]
        elif str(int(float(key_u))) in id_to_new:  # numeric string
            found_u = id_to_new[str(int(float(key_u)))]
        elif str(key_u) in id_to_new:
            found_u = id_to_new[str(key_u)]
        if key_v in id_to_new:
            found_v = id_to_new[key_v]
        elif str(int(float(key_v))) in id_to_new:
            found_v = id_to_new[str(int(float(key_v)))]
        elif str(key_v) in id_to_new:
            found_v = id_to_new[str(key_v)]
        if found_u is None or found_v is None:
            continue
        remapped.append((found_u, found_v, float(w)))

    if len(remapped) == 0:
        logging.warning(f"No valid remapped edges for {city_name}")
    remapped = np.array(remapped)

    n = len(orig_ids)

    # compute or load sdm
    sdm_file = root / "directed_shortest_distance_matrix.npy"
    if sdm_file.exists():
        logging.info(f"Loading existing sdm from {sdm_file}")
        sdm = np.load(sdm_file)
        np.save(out_pre / "preprocessed_sdm.npy", sdm.astype(float))
    else:
        logging.info(f"Computing shortest-distance matrix for {city_name} (n={n})")
        # Auto-enable optimizations for large graphs
        if n > 50000:
            logging.info(f"Large graph detected (n={n}), enabling optimizations...")
            if use_sparse is False:  # Only if not explicitly set
                use_sparse = True
            if batch_size is None:
                batch_size = max(500, n // 200)  # Smaller batches for very large graphs
            if n_jobs is None:
                n_jobs = min(cpu_count(), 16)  # More workers for large graphs
        
        sdm = build_sdm(n, remapped, parallel=parallel, n_jobs=n_jobs, 
                       batch_size=batch_size, use_sparse=use_sparse)
        
        # Convert sparse to dense if needed for saving
        if hasattr(sdm, 'toarray'):
            logging.info("Converting sparse SDM to dense for saving...")
            try:
                sdm_dense = sdm.toarray()
                mask = (sdm_dense == 0) & (np.arange(n)[:, None] != np.arange(n))  # Not diagonal
                sdm_dense[mask] = np.inf
                sdm = sdm_dense
            except MemoryError:
                logging.error(f"Memory error converting sparse to dense for {city_name} (n={n})")
                logging.error("SDM is too large to save as dense .npy. Training might fail.")
                # 如果内存实在不够，建议保存为 npz 格式，但需要同步修改训练端的加载代码
                # np.savez_compressed(out_pre / "preprocessed_sdm.npz", data=sdm) 
                raise MemoryError("Insufficient RAM to convert SDM to dense format for training.")

        # 确保 sdm 最终是 numpy ndarray 格式
        if not isinstance(sdm, np.ndarray):
            sdm = np.array(sdm)
        
        np.save(out_pre / "preprocessed_sdm.npy", sdm.astype(np.float32))  # Use float32 to save memory

    # indices: pairs with finite non-zero distances
    finite_mask = np.isfinite(sdm) & (sdm != 0)
    inds = np.argwhere(finite_mask)

    # try embeddings
    emb = try_load_embedding(root, orig_ids)
    if emb is not None:
        np.save(out_pre / "preprocessed_embed.npy", emb.astype(float))
        logging.info(f"Saved embeddings shape {emb.shape}")

    # save node coords raw and normalized
    np.save(out_pre / "preprocessed_node_long_lat_origin.npy", coords.astype(float))
    coords_norm = normalize_coords(coords)
    np.save(out_pre / "preprocessed_node_long_lat.npy", coords_norm.astype(float))

    np.save(out_pre / "preprocessed_indices.npy", inds.astype(np.int32))

    # landmarks
    try:
        from distnet_model import farthest_selection

        num_landmarks = max(int(n * 0.01), 20)
        lm = farthest_selection(coords, num_landmarks)
        LM_pairs = []
        for i in range(len(lm)):
            for j in range(len(lm)):
                if np.isfinite(sdm[lm[i], lm[j]]) and sdm[lm[i], lm[j]] != 0:
                    LM_pairs.append((lm[i], lm[j]))
        # ensure LM file is always written (possibly empty with shape (0,2))
        if LM_pairs:
            arr = np.array(LM_pairs, dtype=int)
        else:
            arr = np.empty((0, 2), dtype=int)
        np.save(out_pre / "preprocessed_LM_indices.npy", arr)
    except Exception:
        logging.info("farthest_selection not available or failed; saving empty LM indices")
        # write empty (0,2) array so downstream code can always load the file
        np.save(out_pre / "preprocessed_LM_indices.npy", np.empty((0, 2), dtype=int))

    # also save remapped edge file for convenience
    edges_out = out_pre / "preprocessed_edges.csv"
    if remapped.size:
        pd.DataFrame(remapped, columns=["Node_Start", "Node_End", "Length"]).to_csv(edges_out, index=False)
    logging.info(f"Saved preprocessed outputs to {out_pre}")


def preprocess_all(data_dir: Path, cities: list[str] | None = None, 
                   parallel: bool = True, n_jobs: int | None = None, 
                   batch_size: int | None = None, use_sparse: bool = False):
    """
    Main entry point to preprocess one or more cities.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    # If --cities given, only process those; otherwise auto-discover
    if cities:
        for city in cities:
            entry = data_dir / city
            if entry.exists() and entry.is_dir():
                node = find_node_file(entry)
                edge = find_edges_file(entry)
                if node and edge:
                    process_city(entry, city, data_dir, 
                                parallel=parallel, n_jobs=n_jobs,
                                batch_size=batch_size, use_sparse=use_sparse)
                else:
                    logging.warning(f"City directory {entry} missing node/edge files; skipping")
            else:
                # try to find standalone files matching prefix at root and create temp dir
                node_candidates = list(data_dir.glob(f"*{city}*node*.txt")) + list(data_dir.glob(f"*{city}*nodes*.txt"))
                edge_candidates = list(data_dir.glob(f"*{city}*edge*.txt")) + list(data_dir.glob(f"*{city}*edges*.txt"))
                if node_candidates and edge_candidates:
                    tmpdir = data_dir / city
                    tmpdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(node_candidates[0], tmpdir / node_candidates[0].name)
                    shutil.copy(edge_candidates[0], tmpdir / edge_candidates[0].name)
                    process_city(tmpdir, city, data_dir,
                                parallel=parallel, n_jobs=n_jobs,
                                batch_size=batch_size, use_sparse=use_sparse)
                else:
                    logging.warning(f"No node/edge files found for city '{city}'; skipping")
    else:
        # Cities to process: directories under data_dir that contain nodes/edges, plus
        # standalone files like chengdu_node-mod.txt (we map by prefix)
        # first handle subdirectories
        for entry in sorted(data_dir.iterdir()):
            if entry.name == "pre":
                continue
            if entry.is_dir():
                # check for nodes/edges inside
                node = find_node_file(entry)
                edge = find_edges_file(entry)
                if node and edge:
                    process_city(entry, entry.name, data_dir,
                                parallel=parallel, n_jobs=n_jobs,
                                batch_size=batch_size, use_sparse=use_sparse)

        # handle standalone city files at data_dir root (e.g., chengdu_node-mod.txt)
        # group by prefix before first underscore or hyphen
        files = list(data_dir.glob("*node*.txt")) + list(data_dir.glob("*nodes*.txt"))
        for f in files:
            prefix = f.stem.split("_")[0].split("-")[0]
            city_root = data_dir / prefix
            if not city_root.exists():
                # create a temp directory to host the file for processing
                tmpdir = data_dir / prefix
                tmpdir.mkdir(parents=True, exist_ok=True)
                # copy file into tmpdir
                shutil.copy(f, tmpdir / f.name)
                # also try copying edges if a matching edges file exists
                candidates = list(data_dir.glob(f"*{prefix}*edge*.txt")) + list(data_dir.glob(f"*{prefix}*links*.txt"))
                for c in candidates:
                    shutil.copy(c, tmpdir / c.name)
                process_city(tmpdir, prefix, data_dir,
                            parallel=parallel, n_jobs=n_jobs,
                            batch_size=batch_size, use_sparse=use_sparse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data", help="Path to data folder containing city subfolders or city files")
    p.add_argument("--cities", default=None, help="Comma-separated list of city names to process (e.g. harbin,porto). If omitted, process all detected cities.")
    p.add_argument("--parallel", action="store_true", default=True, help="Use parallel computation for SDM (default: True)")
    p.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel computation")
    p.add_argument("--n-jobs", type=int, default=None, help="Number of parallel workers (default: auto, max 8 for small graphs, 16 for large)")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for processing nodes (default: auto based on graph size)")
    p.add_argument("--use-sparse", action="store_true", help="Use sparse matrix storage (saves memory for large graphs)")
    args = p.parse_args()

    cities_list = [c.strip() for c in args.cities.split(",") if c.strip()] if args.cities else None

    preprocess_all(Path(args.data_dir), cities=cities_list, 
                   parallel=args.parallel, n_jobs=args.n_jobs,
                   batch_size=args.batch_size, use_sparse=args.use_sparse)


if __name__ == "__main__":
    main()

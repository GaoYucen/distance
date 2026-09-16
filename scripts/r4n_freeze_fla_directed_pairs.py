"""Freeze the user-authorized DIMACS-FLA native-directed graph and 500K uniform ordered-OD pair workload.

This stage intentionally computes NO shortest-path labels and runs NO model. It freezes the native-directed
largest strongly connected component, deterministic coordinate projection, edge provenance, and the exact
400K/50K/50K ordered pair split before any FLA model evaluation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_dimacs_gr(path: Path):
    n = m = None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("p "):
                p = line.split()
                if len(p) >= 4 and p[1] == "sp":
                    n, m = int(p[2]), int(p[3])
                    break
    if n is None or m is None:
        raise RuntimeError("DIMACS graph header not found")
    src = np.empty(m, dtype=np.int32)
    dst = np.empty(m, dtype=np.int32)
    weight = np.empty(m, dtype=np.int32)
    i = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("a "):
                continue
            _, u, v, w = line.split()[:4]
            if i >= m:
                raise RuntimeError("more arcs than DIMACS header")
            src[i] = int(u) - 1
            dst[i] = int(v) - 1
            weight[i] = int(w)
            i += 1
    if i != m:
        raise RuntimeError(f"arc count mismatch: header={m}, parsed={i}")
    return n, src, dst, weight


def read_dimacs_coords(path: Path, n: int):
    lon = np.full(n, np.nan, dtype=np.float64)
    lat = np.full(n, np.nan, dtype=np.float64)
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, node, x, y = line.split()[:4]
            idx = int(node) - 1
            lon[idx] = float(x) / 1_000_000.0
            lat[idx] = float(y) / 1_000_000.0
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise RuntimeError("missing/nonfinite DIMACS coordinates")
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = transformer.transform(lon, lat)
    coords = np.column_stack((np.asarray(x), np.asarray(y))).astype(np.float32)
    if not np.isfinite(coords).all():
        raise RuntimeError("nonfinite EPSG:5070 coordinates")
    return coords


def unique_min_arcs(src: np.ndarray, dst: np.ndarray, weight: np.ndarray, n: int):
    key = src.astype(np.int64) * np.int64(n) + dst.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    src = src[order]
    dst = dst[order]
    weight = weight[order]
    start = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(start)
    minw = np.minimum.reduceat(weight, starts)
    return src[starts], dst[starts], minw.astype(np.int32)


def sample_unique_ordered_pair_codes(n: int, k: int, seed: int):
    total = int(n) * int(n - 1)
    if k > total:
        raise ValueError("requested more ordered non-self pairs than exist")
    rng = np.random.default_rng(seed)
    seen: set[int] = set()
    out = np.empty(k, dtype=np.int64)
    filled = 0
    while filled < k:
        batch = min(max((k - filled) * 2, 10000), 1_000_000)
        vals = rng.integers(0, total, size=batch, dtype=np.int64)
        for q in vals:
            z = int(q)
            if z in seen:
                continue
            seen.add(z)
            out[filled] = z
            filled += 1
            if filled == k:
                break
    return out


def decode_codes(codes: np.ndarray, n: int):
    u = codes // (n - 1)
    r = codes % (n - 1)
    v = r + (r >= u)
    return np.column_stack((u, v)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", type=Path, required=True)
    ap.add_argument("--co", type=Path, required=True)
    ap.add_argument("--graph-output", type=Path, required=True)
    ap.add_argument("--pairs-output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026091601)
    ap.add_argument("--num-pairs", type=int, default=500000)
    args = ap.parse_args()
    for p in (args.graph_output, args.pairs_output, args.report):
        if p.exists():
            raise FileExistsError(p)

    started = time.perf_counter()
    n_raw, src, dst, weight = read_dimacs_gr(args.gr)
    coords_raw = read_dimacs_coords(args.co, n_raw)

    adjacency = csr_matrix((np.ones(len(src), dtype=np.uint8), (src, dst)), shape=(n_raw, n_raw))
    ncomp, labels = connected_components(adjacency, directed=True, connection="strong", return_labels=True)
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    keep_nodes = np.flatnonzero(labels == largest_label).astype(np.int64)
    n = len(keep_nodes)
    old_to_new = np.full(n_raw, -1, dtype=np.int32)
    old_to_new[keep_nodes] = np.arange(n, dtype=np.int32)
    mask = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
    s = old_to_new[src[mask]]
    d = old_to_new[dst[mask]]
    w = weight[mask]
    s, d, w = unique_min_arcs(s, d, w, n)
    if np.any(s == d):
        nons = s != d
        s, d, w = s[nons], d[nons], w[nons]
    if np.any(w < 0):
        raise RuntimeError("negative FLA arc weight")

    coords = coords_raw[keep_nodes]
    original_node_ids = keep_nodes + 1
    codes = sample_unique_ordered_pair_codes(n, args.num_pairs, args.seed)
    pairs = decode_codes(codes, n)
    if len(np.unique(codes)) != args.num_pairs or np.any(pairs[:, 0] == pairs[:, 1]):
        raise AssertionError("pair freeze invariant failed")
    train_pairs = pairs[:400000]
    validation_pairs = pairs[400000:450000]
    test_pairs = pairs[450000:500000]
    if not (len(train_pairs) == 400000 and len(validation_pairs) == 50000 and len(test_pairs) == 50000):
        raise AssertionError("authorized split invariant failed")

    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.graph_output, src=s, dst=d, weight=w, coordinates=coords,
                        original_node_ids=original_node_ids.astype(np.int64))
    np.savez_compressed(args.pairs_output, train_pairs=train_pairs, validation_pairs=validation_pairs,
                        test_pairs=test_pairs, coordinates=coords,
                        original_node_ids=original_node_ids.astype(np.int64), seed=np.int64(args.seed))

    report = {
        "status": "frozen_pairs_before_labels_or_model_evaluation",
        "classification": "user-authorized DIMACS-FLA native-directed confirmation workload",
        "raw_gr_sha256": sha256(args.gr),
        "raw_co_sha256": sha256(args.co),
        "raw_nodes": int(n_raw),
        "raw_arcs": int(len(src)),
        "strong_component_count": int(ncomp),
        "lscc_rule": "largest strongly connected component; ties resolved by numpy argmax smallest label",
        "lscc_nodes": int(n),
        "lscc_fraction": float(n / n_raw),
        "lscc_directed_arcs_after_parallel_min_dedup": int(len(s)),
        "coordinate_policy": "DIMACS lon/lat integer microdegrees -> WGS84 degrees -> EPSG:5070 meters, matching Survey DIMACS coordinate policy",
        "pair_policy": "500K unique uniform ordered non-self pairs on native-directed FLA LSCC; 400K/50K/50K by frozen order",
        "pair_seed": int(args.seed),
        "splits": [400000, 50000, 50000],
        "graph_output": str(args.graph_output),
        "graph_sha256": sha256(args.graph_output),
        "pairs_output": str(args.pairs_output),
        "pairs_sha256": sha256(args.pairs_output),
        "labels_computed": False,
        "model_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("R4N_FLA_PAIRS_FROZEN", json.dumps(report))


if __name__ == "__main__":
    main()

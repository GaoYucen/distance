"""Build the frozen random32 directed-landmark table on a sparse native-directed graph.

This is the sparse-graph equivalent of slicing 32 rows/columns from an all-pairs matrix. It does not
change landmark selection or the R4M index budget: 32 forward + 32 reverse float32 scalars per node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

LANDMARK_SEED = 20260914


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("directed landmark output exists")
    g = np.load(args.graph)
    z = np.load(args.workload)
    coords = np.asarray(g["coordinates"])
    n = len(coords)
    if len(z["coordinates"]) != n:
        raise RuntimeError("graph/workload node count mismatch")
    A = csr_matrix((np.asarray(g["weight"], dtype=np.float64),
                    (np.asarray(g["src"], dtype=np.int64), np.asarray(g["dst"], dtype=np.int64))),
                   shape=(n, n))
    train = np.asarray(z["train"])
    train_nodes = np.unique(train[:, :2].astype(np.int64))
    rng = np.random.default_rng(LANDMARK_SEED)
    landmarks = np.sort(rng.choice(train_nodes, size=32, replace=False))
    started = time.perf_counter()
    fwd = dijkstra(A, directed=True, indices=landmarks)
    rev = dijkstra(A.T, directed=True, indices=landmarks)
    elapsed = time.perf_counter() - started
    if fwd.shape != (32, n) or rev.shape != (32, n):
        raise RuntimeError((fwd.shape, rev.shape, n))
    if not np.isfinite(fwd).all() or not np.isfinite(rev).all():
        raise RuntimeError("nonfinite directed landmark distances on frozen strongly connected graph")
    node = np.concatenate((fwd.T, rev.T), axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=node, landmarks=landmarks.astype(np.int64))
    report = {
        "status": "completed",
        "classification": "frozen random32 directed landmark index; sparse exact-Dijkstra construction",
        "graph_sha256": sha256(args.graph),
        "workload_sha256": sha256(args.workload),
        "landmark_seed": LANDMARK_SEED,
        "landmarks": landmarks.tolist(),
        "node_count": int(n),
        "index_scalars_per_node": 64,
        "index_bytes_per_node_float32": 256,
        "output_sha256": sha256(args.output),
        "build_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("R4N_SPARSE_DIRECTED_LANDMARK_INDEX_COMPLETE", json.dumps(report))


if __name__ == "__main__":
    main()

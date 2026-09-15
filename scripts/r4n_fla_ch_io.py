"""Prepare/finalize exact RoutingKit-CH labels for the already-frozen DIMACS-FLA workload.

`prepare` converts immutable frozen NPZ graph/pairs into simple uint32 binary arrays. Query order is:
500K primary ordered pairs (400K/50K/50K), then the 50K reversed test pairs used only for asymmetry metrics.
`finalize` consumes exact CH distances, independently verifies a deterministic subset with SciPy directed
Dijkstra, and only then materializes the labeled 400K/50K/50K workload. No model is run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare(graph_path: Path, pairs_path: Path, out_dir: Path):
    if out_dir.exists():
        raise FileExistsError(out_dir)
    g = np.load(graph_path)
    p = np.load(pairs_path)
    src = np.asarray(g["src"], dtype=np.int64)
    dst = np.asarray(g["dst"], dtype=np.int64)
    weight = np.asarray(g["weight"], dtype=np.int64)
    coords = np.asarray(g["coordinates"])
    n = len(coords)
    if src.shape != dst.shape or src.shape != weight.shape:
        raise RuntimeError("graph array shape mismatch")
    if np.any(src < 0) or np.any(src >= n) or np.any(dst < 0) or np.any(dst >= n):
        raise RuntimeError("arc endpoint outside node range")
    if np.any(weight < 0) or np.any(weight > np.iinfo(np.uint32).max):
        raise RuntimeError("arc weight outside uint32")
    train = np.asarray(p["train_pairs"], dtype=np.int64)
    val = np.asarray(p["validation_pairs"], dtype=np.int64)
    test = np.asarray(p["test_pairs"], dtype=np.int64)
    if (len(train), len(val), len(test)) != (400000, 50000, 50000):
        raise RuntimeError((len(train), len(val), len(test)))
    primary = np.vstack((train, val, test))
    reverse_test = test[:, ::-1].copy()
    queries = np.vstack((primary, reverse_test))
    if np.any(queries < 0) or np.any(queries >= n):
        raise RuntimeError("query endpoint outside node range")
    out_dir.mkdir(parents=True)
    arrays = {
        "tail.u32": src.astype(np.uint32),
        "head.u32": dst.astype(np.uint32),
        "weight.u32": weight.astype(np.uint32),
        "query_src.u32": queries[:, 0].astype(np.uint32),
        "query_dst.u32": queries[:, 1].astype(np.uint32),
    }
    file_meta = {}
    for name, arr in arrays.items():
        path = out_dir / name
        arr.tofile(path)
        file_meta[name] = {"count": int(arr.size), "sha256": sha256(path)}
    meta = {
        "status": "prepared_before_exact_labels",
        "node_count": int(n),
        "arc_count": int(len(src)),
        "primary_query_count": 500000,
        "reverse_test_query_count": 50000,
        "total_query_count": 550000,
        "query_order": "train400k,val50k,test50k,reverse(test)50k",
        "graph_npz": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "pairs_npz": str(pairs_path),
        "pairs_sha256": sha256(pairs_path),
        "files": file_meta,
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("R4N_FLA_CH_PREPARED", json.dumps(meta))


def finalize(graph_path: Path, pairs_path: Path, io_dir: Path, distance_path: Path,
             workload_out: Path, report_out: Path, check_count: int):
    if workload_out.exists() or report_out.exists():
        raise FileExistsError("final FLA output already exists")
    meta = json.loads((io_dir / "prepare_meta.json").read_text())
    if meta["graph_sha256"] != sha256(graph_path) or meta["pairs_sha256"] != sha256(pairs_path):
        raise RuntimeError("frozen input hash changed after CH preparation")
    for name, rec in meta["files"].items():
        path = io_dir / name
        if rec["sha256"] != sha256(path):
            raise RuntimeError(f"CH input hash mismatch: {name}")
    dist = np.fromfile(distance_path, dtype=np.uint32)
    if len(dist) != 550000:
        raise RuntimeError((len(dist), 550000))
    if np.any(dist == np.iinfo(np.uint32).max):
        raise RuntimeError("unreachable/invalid CH distance encountered")
    p = np.load(pairs_path)
    g = np.load(graph_path)
    train_pairs = np.asarray(p["train_pairs"], dtype=np.int64)
    val_pairs = np.asarray(p["validation_pairs"], dtype=np.int64)
    test_pairs = np.asarray(p["test_pairs"], dtype=np.int64)
    primary_pairs = np.vstack((train_pairs, val_pairs, test_pairs))
    primary_dist = dist[:500000].astype(np.float64)
    reverse_test = dist[500000:].astype(np.float64)
    if np.any(primary_dist <= 0) or np.any(reverse_test <= 0):
        raise RuntimeError("non-positive non-self FLA distance")

    n = len(g["coordinates"])
    A = csr_matrix((np.asarray(g["weight"], dtype=np.float64),
                    (np.asarray(g["src"], dtype=np.int64), np.asarray(g["dst"], dtype=np.int64))),
                   shape=(n, n))
    rng = np.random.default_rng(2026091602)
    idx = np.sort(rng.choice(len(primary_pairs), size=check_count, replace=False))
    selected = primary_pairs[idx]
    unique_src = np.unique(selected[:, 0])
    D = dijkstra(A, directed=True, indices=unique_src)
    src_pos = {int(s): i for i, s in enumerate(unique_src)}
    errors = []
    for qi, (u, v) in zip(idx, selected):
        exact = float(D[src_pos[int(u)], int(v)])
        got = float(primary_dist[int(qi)])
        errors.append(abs(exact - got))
    max_err = float(max(errors, default=0.0))
    if max_err != 0.0:
        raise AssertionError(f"RoutingKit CH vs SciPy directed Dijkstra mismatch: {max_err}")

    train = np.column_stack((train_pairs, primary_dist[:400000]))
    val = np.column_stack((val_pairs, primary_dist[400000:450000]))
    test = np.column_stack((test_pairs, primary_dist[450000:500000]))
    workload_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        workload_out,
        train=train,
        validation=val,
        test=test,
        test_reverse_distances=reverse_test,
        coordinates=np.asarray(g["coordinates"], dtype=np.float32),
        original_node_ids=np.asarray(g["original_node_ids"], dtype=np.int64),
    )
    alpha = np.abs(test[:, 2] - reverse_test) / ((test[:, 2] + reverse_test) / 2.0)
    report = {
        "status": "completed_exact_labels_verified",
        "classification": "user-authorized native-directed DIMACS-FLA confirmation workload; pairs frozen before labels/models",
        "oracle": "RoutingKit ContractionHierarchy exact directed shortest-path queries",
        "routingkit_commit": "54d49bb0cdea56dde182357522e4e86a03c57852",
        "graph_sha256": sha256(graph_path),
        "pairs_sha256": sha256(pairs_path),
        "ch_input_meta_sha256": sha256(io_dir / "prepare_meta.json"),
        "distance_binary_sha256": sha256(distance_path),
        "workload_sha256": sha256(workload_out),
        "node_count": int(n),
        "splits": [400000, 50000, 50000],
        "independent_scipy_directed_dijkstra_checks": int(check_count),
        "independent_check_max_abs_distance": max_err,
        "test_alpha_ge_20pct_fraction": float(np.mean(alpha >= 0.2)),
        "test_mean_alpha_percent": float(100.0 * np.mean(alpha)),
        "model_evaluated": False,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print("R4N_FLA_LABELS_FINALIZED", json.dumps(report))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--graph", type=Path, required=True)
    f.add_argument("--pairs", type=Path, required=True)
    f.add_argument("--io-dir", type=Path, required=True)
    f.add_argument("--distances", type=Path, required=True)
    f.add_argument("--workload-out", type=Path, required=True)
    f.add_argument("--report-out", type=Path, required=True)
    f.add_argument("--check-count", type=int, default=64)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.graph, args.pairs, args.out_dir)
    else:
        finalize(args.graph, args.pairs, args.io_dir, args.distances,
                 args.workload_out, args.report_out, args.check_count)


if __name__ == "__main__":
    main()

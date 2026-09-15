"""Export Survey-compatible undirected graph-side assets for frozen native-directed workloads.

This helper is ONLY for Survey-style directed-target adaptations. Ground-truth query labels/splits are
copied verbatim from the frozen native-directed R4N workloads. The graph-side representation deliberately
keeps the Survey benchmark's original undirected preprocessing bias: every native directed arc is collapsed
onto an undirected pair, retaining the minimum observed weight. No new directional capacity is introduced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_shenzhen_edges(workload, raw_csv: Path):
    original = workload["original_node_ids"].astype(np.int64)
    n = len(original)
    maxid = int(original.max())
    mapping = np.full(maxid + 1, -1, dtype=np.int64)
    mapping[original] = np.arange(n, dtype=np.int64)
    directed = {}
    with raw_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            u0, v0 = int(row["Origin"]), int(row["Destination"])
            if u0 > maxid or v0 > maxid:
                continue
            u, v = int(mapping[u0]), int(mapping[v0])
            if u < 0 or v < 0 or u == v:
                continue
            w = float(row["Length"])
            key = (u, v)
            if key not in directed or w < directed[key]:
                directed[key] = w
    return n, directed, sha256(raw_csv)


def load_chengdu_edges(workload, link_csv: Path):
    n = len(workload["coordinates"])
    directed = {}
    with link_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            u, v = int(row["Node_Start"]), int(row["Node_End"])
            if not (0 <= u < n and 0 <= v < n):
                raise RuntimeError(f"Chengdu node id outside frozen workload indexing: {(u, v, n)}")
            if u == v:
                continue
            w = float(row["Length"])
            key = (u, v)
            if key not in directed or w < directed[key]:
                directed[key] = w
    return n, directed, sha256(link_csv)


def collapse_undirected(directed: dict[tuple[int, int], float]):
    undirected = {}
    for (u, v), w in directed.items():
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in undirected or w < undirected[key]:
            undirected[key] = w
    return undirected


def write_queries(out_dir: Path, prefix: str, workload_path: Path, z):
    files = {}
    for key, suffix in (("train", "train"), ("validation", "val"), ("test", "test")):
        a = np.asarray(z[key])
        if a.ndim != 2 or a.shape[1] < 3:
            raise RuntimeError((key, a.shape))
        p = out_dir / f"{prefix}_{suffix}.queries.npz"
        np.savez_compressed(
            p,
            src=a[:, 0].astype(np.int64) + 1,
            dst=a[:, 1].astype(np.int64) + 1,
            dist=a[:, 2].astype(np.float32),
        )
        files[key] = {"rows": int(len(a)), "sha256": sha256(p)}
    sets = {k: set(map(tuple, np.asarray(z[k])[:, :2].astype(np.int64))) for k in ("train", "validation", "test")}
    overlaps = {
        "train_val": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["validation"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(overlaps)
    return {
        "source_workload": str(workload_path),
        "source_workload_sha256": sha256(workload_path),
        "files": files,
        "ordered_pair_overlaps": overlaps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shenzhen", "chengdu"], required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--raw-edges", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    z = np.load(args.workload)
    coords = np.asarray(z["coordinates"], dtype=np.float64)
    if not np.isfinite(coords).all():
        raise RuntimeError("nonfinite coordinates")
    if args.dataset == "shenzhen":
        n, directed, raw_sha = load_shenzhen_edges(z, args.raw_edges)
    else:
        n, directed, raw_sha = load_chengdu_edges(z, args.raw_edges)
    if len(coords) != n:
        raise RuntimeError((len(coords), n))
    undirected = collapse_undirected(directed)
    if not undirected:
        raise RuntimeError("empty Survey-compatible graph")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    edge_path = args.output_dir / f"{args.prefix}.edges"
    node_path = args.output_dir / f"{args.prefix}.nodes"
    with edge_path.open("w", newline="") as f:
        w = csv.writer(f)
        for (u, v), d in sorted(undirected.items()):
            w.writerow((u + 1, v + 1, float(d)))
    with node_path.open("w", newline="") as f:
        w = csv.writer(f)
        for i, (x, y) in enumerate(coords):
            w.writerow((i + 1, float(x), float(y)))
    qdir = args.output_dir / "directed_target_500k"
    qdir.mkdir()
    qmeta = write_queries(qdir, args.prefix, args.workload, z)
    meta = {
        "status": "completed",
        "classification": "Survey-style directed-target adaptation; native-directed labels with original undirected graph-side inductive bias",
        "dataset": args.dataset,
        "prefix": args.prefix,
        "node_count": int(n),
        "native_directed_arc_count": int(len(directed)),
        "survey_undirected_edge_count": int(len(undirected)),
        "undirected_collapse_rule": "for each unordered pair present in >=1 native arc direction, retain minimum observed native arc weight",
        "raw_edge_source": str(args.raw_edges),
        "raw_edge_sha256": raw_sha,
        "edges_sha256": sha256(edge_path),
        "nodes_sha256": sha256(node_path),
        "queries": qmeta,
        "directional_capacity_added": False,
    }
    (args.output_dir / "DIRECTED_TARGET_ASSET_META.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("R4N_SURVEY_MATCHED_ASSETS", json.dumps(meta))


if __name__ == "__main__":
    main()

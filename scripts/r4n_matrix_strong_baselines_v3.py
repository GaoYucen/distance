"""Deterministic numerical fix for r4n_matrix_strong_baselines_v2.

The v2 baseline runner used central float32 landmark values directly for ALT32's
certificate audit. R4E/R4M's frozen certified-index policy is instead: store the
64 landmark scalars as float32, then outward-round each stored scalar by one
float32 ULP before decoding L/U. This wrapper replaces only that bound decoder
and delegates all model/data/seed/time-budget logic to v2 unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.r4n_matrix_strong_baselines_v2 as base


def directed_bounds_outward(node: np.ndarray, pairs: np.ndarray):
    pairs = np.asarray(pairs, dtype=np.int64)
    u, v = pairs.T
    x = np.asarray(node, dtype=np.float32)
    k = x.shape[1] // 2

    lo = np.nextafter(x, np.float32(-np.inf), dtype=np.float32).astype(np.float64)
    hi = np.nextafter(x, np.float32(np.inf), dtype=np.float32).astype(np.float64)
    flo, fhi = lo[:, :k], hi[:, :k]
    tlo, thi = lo[:, k:], hi[:, k:]

    lower_safe = np.concatenate(
        (flo[v] - fhi[u], tlo[u] - thi[v]), axis=1
    )
    upper_safe = thi[u] + fhi[v]
    L = np.maximum(0.0, lower_safe.max(axis=1))
    U = upper_safe.min(axis=1)
    if np.any(U < L):
        raise AssertionError(
            ('invalid landmark bounds after outward rounding', float(np.max(L - U)))
        )
    return L.astype(np.float64), U.astype(np.float64)


base.directed_bounds = directed_bounds_outward

if __name__ == '__main__':
    base.main()

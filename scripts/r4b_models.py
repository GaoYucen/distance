"""Fixed-budget R4B decoder study. IQE uses the authors' pinned implementation.
All models store 64 float32 scalars per node. The seed axis is vectorized,
with independent parameters, objectives, Adam states and checkpoint selection.
No GNN or query-dependent encoder is used.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

UPSTREAM_SHA = '0365394d4ecbd614f38775787af724d8b142c9d2'
UPSTREAM = Path(os.environ.get('R4B_TORCHQMET', '/workspace/.server-control/sources/distance/torch-quasimetric-0365394'))
if UPSTREAM.is_dir():
    sys.path.insert(0, str(UPSTREAM))
from torchqmet.iqe import iqe

MODES = ('L1', 'T1', 'P64', 'B2', 'B4', 'B8', 'Shared-L1', 'MRN-L2', 'IQE-maxmean', 'IQE-sum')
SEEDS = (42, 99, 1234)


def digest_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def random_tables(n: int, seeds=SEEDS) -> torch.Tensor:
    return torch.stack([torch.randn(n, 64, generator=torch.Generator().manual_seed(s)) * .1 for s in seeds])


def components(x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    delta = y - x
    if mode == 'L1':
        return delta.abs().sum(-1, keepdim=True)
    if mode in ('T1', 'B2', 'B4', 'B8'):
        k = 1 if mode == 'T1' else int(mode[1:])
        a = delta.unflatten(-1, (k, 64 // k))
        return a[..., :-1].abs().sum(-1) + a[..., -1]
    if mode == 'P64':
        return delta
    if mode in ('Shared-L1', 'MRN-L2'):
        r = 56 if mode == 'Shared-L1' else 32
        sym = delta[..., :r].abs().sum(-1, keepdim=True) if mode == 'Shared-L1' else torch.linalg.vector_norm(delta[..., :r], dim=-1, keepdim=True)
        return torch.cat((sym, sym + delta[..., r:]), dim=-1)
    if mode.startswith('IQE-'):
        return iqe(x.unflatten(-1, (8, 8)), y.unflatten(-1, (8, 8)))
    raise ValueError(mode)


def reduce_components(c: torch.Tensor, mode: str, alpha: torch.Tensor) -> torch.Tensor:
    if mode == 'IQE-sum':
        return c.sum(-1)
    if mode == 'IQE-maxmean':
        return torch.lerp(c.mean(-1), c.max(-1).values, alpha.sigmoid().reshape(-1, 1, 1))
    return c.max(-1).values.relu()


class SeedBatch(nn.Module):
    def __init__(self, n: int, mode: str, seeds=SEEDS):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.table = nn.Parameter(random_tables(n, seeds))
        if mode == 'IQE-maxmean':
            self.raw_alpha = nn.Parameter(torch.full((len(seeds),), -1.))
        else:
            self.register_buffer('raw_alpha', torch.full((len(seeds),), -1.))
        self.register_buffer('calibration', torch.ones(len(seeds)))

    def forward(self, return_components=False):
        c = components(self.table[:, :, None, :], self.table[:, None, :, :], self.mode)
        pred = reduce_components(c, self.mode, self.raw_alpha) * self.calibration[:, None, None]
        return (pred, c) if return_components else pred

    @torch.no_grad()
    def calibrate(self, mask: torch.Tensor):
        mean = self()[:, mask].mean(-1)
        if not bool(torch.isfinite(mean).all() and (mean > 0).all()):
            raise ValueError('Invalid initial prediction mean')
        self.calibration.copy_(1 / mean)
        torch.testing.assert_close(self()[:, mask].mean(-1), torch.ones_like(mean), atol=2e-6, rtol=2e-6)


def numpy_components(table: np.ndarray, mode: str) -> np.ndarray:
    """Independent float64 decoder; interval merging, not endpoint events, for IQE."""
    a = np.asarray(table, dtype=np.float64)
    x, y = a[:, None, :], a[None, :, :]
    d = y - x
    if mode == 'L1':
        return np.abs(d).sum(-1, keepdims=True)
    if mode in ('T1', 'B2', 'B4', 'B8'):
        k = 1 if mode == 'T1' else int(mode[1:])
        block = d.reshape(len(a), len(a), k, 64 // k)
        return np.abs(block[..., :-1]).sum(-1) + block[..., -1]
    if mode == 'P64':
        return d
    if mode in ('Shared-L1', 'MRN-L2'):
        r = 56 if mode == 'Shared-L1' else 32
        sym = np.abs(d[..., :r]).sum(-1, keepdims=True) if mode == 'Shared-L1' else np.sqrt(np.square(d[..., :r]).sum(-1, keepdims=True))
        return np.concatenate((sym, sym + d[..., r:]), axis=-1)
    if mode.startswith('IQE-'):
        lo, end = np.broadcast_arrays(x.reshape(len(a), 1, 8, 8), y.reshape(1, len(a), 8, 8))
        end = np.maximum(lo, end)
        order = np.argsort(lo, axis=-1, kind='stable')
        left = np.take_along_axis(lo, order, axis=-1)
        right = np.take_along_axis(end, order, axis=-1)
        covered_end = np.maximum.accumulate(right, axis=-1)
        previous = np.concatenate((np.full_like(right[..., :1], -np.inf), covered_end[..., :-1]), axis=-1)
        return np.maximum(0., right - np.maximum(left, previous)).sum(-1)
    raise ValueError(mode)


def numpy_decode(table: np.ndarray, mode: str, raw_alpha: float, calibration: float) -> np.ndarray:
    c = numpy_components(table, mode)
    if mode == 'IQE-sum':
        out = c.sum(-1)
    elif mode == 'IQE-maxmean':
        alpha = 1 / (1 + np.exp(-raw_alpha))
        out = (1 - alpha) * c.mean(-1) + alpha * c.max(-1)
    else:
        out = np.maximum(0., c.max(-1))
    return out * calibration


def metrics(pred: np.ndarray, truth: np.ndarray, triples: np.ndarray) -> dict:
    n = len(truth)
    mask = ~np.eye(n, dtype=bool)
    du, pu = truth[mask], pred[mask]
    rel = np.abs(pu - du) / du
    A = (truth - truth.T) / 2
    pa = (pred - pred.T) / 2
    i, j, k = triples.T
    cycle = A[i, j] + A[j, k] + A[k, i]
    pcycle = pa[i, j] + pa[j, k] + pa[k, i]
    asym = np.abs(truth - truth.T) / np.maximum((truth + truth.T) / 2, 1e-300)
    high = mask & (asym >= .2)
    short = mask & (truth <= np.quantile(du, .25))
    return {
        'mre_percent': float(rel.mean() * 100), 'p95_relative_percent': float(np.quantile(rel, .95) * 100),
        'max_relative_percent': float(rel.max() * 100), 'mae': float(np.abs(pu-du).mean()),
        'normalized_mse': float(np.mean(np.square(pu-du)) / du.mean()**2),
        'direction_half_difference_rmse': float(np.sqrt(np.mean(np.square((pa-A)[mask])))),
        'cycle_half_difference_rmse': float(np.sqrt(np.mean((pcycle-cycle)**2))),
        'prediction_cycle_max_abs': float(np.abs(pcycle).max()),
        'overprediction_fraction': float(np.mean(pu > du + 1e-5)),
        'underprediction_fraction': float(np.mean(pu < du - 1e-5)),
        'negative_prediction_fraction': float(np.mean(pu < -1e-7)),
        'high_asymmetry_mre_percent': float((np.abs(pred[high]-truth[high])/truth[high]).mean()*100) if high.any() else None,
        'short_quartile_mre_percent': float((np.abs(pred[short]-truth[short])/truth[short]).mean()*100)
    }


def triangle_check(pred: np.ndarray, tolerance: float) -> dict:
    maximum, count = 0., 0
    for j in range(len(pred)):
        violation = pred - pred[:, j:j+1] - pred[j:j+1, :]
        maximum = max(maximum, float(violation.max()))
        count += int((violation > tolerance).sum())
    return {'checked_triples': int(len(pred)**3), 'max_positive_violation': maximum,
            'violations_above_tolerance': count, 'tolerance': tolerance,
            'passed': count == 0}

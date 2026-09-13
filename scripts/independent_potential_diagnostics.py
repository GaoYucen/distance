"""Sparse least-squares diagnostics for A_uv=(d_uv-d_vu)/2.

This is a model-class diagnostic, NOT a deployable distance predictor.
In particular S_uv used by the oracle reconstruction is a true test label.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr


@dataclass
class PotentialFit:
    h: np.ndarray
    nodes: np.ndarray
    components: np.ndarray
    degree: np.ndarray
    solver: dict
    relative_weighting: bool

    def predict(self, q):
        ids = np.asarray(q[:, :2], dtype=np.int64)
        positions = np.searchsorted(self.nodes, ids)
        clipped = np.minimum(positions, len(self.nodes) - 1)
        known = (positions < len(self.nodes)) & (self.nodes[clipped] == ids)
        u, v = clipped[:, 0], clipped[:, 1]
        identifiable = known.all(axis=1) & (self.degree[u] > 0) & (self.degree[v] > 0) & (self.components[u] == self.components[v])
        pred = self.h[v] - self.h[u]
        pred[~identifiable] = np.nan
        return pred, identifiable


def fit_potential(q, *, relative_weighting=False, tolerance=1e-9, maxiter=2000):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q):
        raise ValueError('Need a nonempty four-column query array.')
    nodes, inv = np.unique(q[:, :2].astype(np.int64), return_inverse=True)
    ids = inv.reshape(-1, 2)
    m, n = len(q), len(nodes)
    B = coo_matrix((np.tile([-1., 1.], m), (np.repeat(np.arange(m), 2), ids.ravel())), shape=(m, n)).tocsr()
    a = (q[:, 2] - q[:, 3]) / 2
    s = (q[:, 2] + q[:, 3]) / 2
    row_scale = 1 / np.maximum(s, 1.0) if relative_weighting else np.ones(m)
    BW = B.multiply(row_scale[:, None]).tocsr()
    answer = lsmr(BW, a * row_scale, atol=tolerance, btol=tolerance, maxiter=maxiter)
    h, status, iterations = answer[0], int(answer[1]), int(answer[2])
    if status not in (0, 1, 2, 4, 5):
        raise RuntimeError(f'LSMR did not converge acceptably: istop={status}, iterations={iterations}')
    adjacency = coo_matrix((np.ones(2 * m), (ids.ravel(), ids[:, ::-1].ravel())), shape=(n, n)).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    degree = np.bincount(ids.ravel(), minlength=n)
    residual = BW @ h - a * row_scale
    normal_residual = float(np.linalg.norm(BW.T @ residual))
    solver = {'method': 'scipy.sparse.linalg.lsmr', 'istop': status,
              'iterations': iterations, 'atol': tolerance, 'btol': tolerance,
              'residual_norm': float(answer[3]), 'normal_residual_norm': normal_residual,
              'num_train_nodes': n, 'num_train_components': int(count),
              'weighting': '1/max(S_uv,1 metre)^2' if relative_weighting else 'uniform'}
    return PotentialFit(h, nodes, labels, degree, solver, relative_weighting)


def _stats(q, predicted_a):
    if len(q) == 0:
        return {'n': 0}
    s = (q[:, 2] + q[:, 3]) / 2
    a = (q[:, 2] - q[:, 3]) / 2
    residual = a - predicted_a
    energy = float(a @ a)
    relative_energy = float(np.sum((a / s)**2))
    ratio = np.abs(a) / s
    oracle_uv, oracle_vu = s + predicted_a, s - predicted_a
    return {
        'n': len(q),
        'asymmetry_definition': 'abs(d_uv-d_vu)/(d_uv+d_vu) = abs(A_uv)/S_uv',
        'asymmetry_mean': float(ratio.mean()),
        'asymmetry_median': float(np.median(ratio)),
        'fraction_asymmetry_ge_0.10': float((ratio >= 0.1).mean()),
        'direction_A_rmse_m': float(np.sqrt(np.mean(a**2))),
        'potential_residual_A_rmse_m': float(np.sqrt(np.mean(residual**2))),
        'potential_explained_A_energy': 1 - float(residual @ residual) / energy if energy > 0 else None,
        'potential_explained_relative_A_energy': 1 - float(np.sum((residual / s)**2)) / relative_energy if relative_energy > 0 else None,
        'symmetric_oracle_min_possible_bidirectional_mse_m2': float(np.mean(a**2)),
        'symmetric_oracle_min_possible_bidirectional_mre': float(np.mean(np.abs(q[:, 2] - q[:, 3]) / (2 * np.maximum(q[:, 2], q[:, 3])))),
        'true_S_plus_fitted_A_oracle_mre': float(np.mean((np.abs(oracle_uv - q[:, 2]) / q[:, 2] + np.abs(oracle_vu - q[:, 3]) / q[:, 3]) / 2)),
        'true_S_plus_fitted_A_oracle_negative_rate': float(np.mean(np.r_[oracle_uv < 0, oracle_vu < 0])),
    }


def summarize_fit(model, q):
    pred, identifiable = model.predict(q)
    s = (q[:, 2] + q[:, 3]) / 2
    ratio = np.abs(q[:, 2] - q[:, 3]) / (q[:, 2] + q[:, 3])
    masks = {'all_identifiable': identifiable,
             'short_S_le_1000m': identifiable & (s <= 1000),
             'medium_S_1000_to_5000m': identifiable & (s > 1000) & (s <= 5000),
             'long_S_gt_5000m': identifiable & (s > 5000),
             'high_asymmetry_ratio_ge_0.10': identifiable & (ratio >= 0.1)}
    return {'query_count': len(q), 'identifiable_count': int(identifiable.sum()),
            'identifiable_fraction': float(identifiable.mean()) if len(q) else 0.,
            'buckets': {name: _stats(q[mask], pred[mask]) for name, mask in masks.items()}}


def symmetric_error_floor(q):
    return _stats(q, np.zeros(len(q)))

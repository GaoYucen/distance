"""R4N matrix-backed strong baseline runner for frozen native-directed workloads.

This script is a protocol-preserving generalization of r4f_directed_strong_baselines.py
from the Jinan development workload to any already-frozen workload with an exact
native-directed distance matrix. It does not define a new baseline family.

Supported evidence:
- ALT32 certified directed-landmark lower-bound reference.
- Dir-CatBoost (our adaptation): Survey CatBoost-style pair features using the same
  32 forward+reverse directed landmarks (64 float32/node) as R4M.
- Dir-LandmarkNN (our adaptation): Survey LandmarkNN architecture/feature recipe
  with the same directed landmark index.

The default final protocol is 300 seconds/model seed and seeds 42,99,1234. A caller
may pass a strict subset of seeds for a bounded screening run; such an output is
explicitly classified as screening and is not a final stochastic-baseline result.
Test labels are loaded only after validation-based model states are frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS_FINAL = (42, 99, 1234)
LANDMARK_SEED = 20260914
BATCH = 16384
LR = 1e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def node_stats(a: np.ndarray):
    mu = a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def encode_numpy(node, coords, pairs, normalized=False):
    pairs = np.asarray(pairs, dtype=np.int64)
    u, v = pairs.T
    land = np.asarray(node, dtype=np.float32)
    xy = np.asarray(coords, dtype=np.float32)
    if normalized:
        lm, ls = node_stats(land)
        cm, cs = node_stats(xy)
        land = (land - lm) / ls
        xy = (xy - cm) / cs
    a, b = land[u], land[v]
    ca, cb = xy[u], xy[v]
    dot = np.sum(a * b, axis=1)
    den = np.maximum(np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-12)
    cos = (dot / den)[:, None].astype(np.float32)
    l1 = np.abs(ca - cb).sum(1, keepdims=True).astype(np.float32)
    l2 = np.sqrt(np.square(ca - cb).sum(1, keepdims=True)).astype(np.float32)
    base = np.concatenate((a, b, ca, cb, cos), axis=1).astype(np.float32)
    return base, l1, l2


def directed_bounds(node: np.ndarray, pairs: np.ndarray):
    pairs = np.asarray(pairs, dtype=np.int64)
    u, v = pairs.T
    k = node.shape[1] // 2
    fwd = node[:, :k]  # d(landmark, x)
    rev = node[:, k:]  # d(x, landmark)
    lower = np.maximum(
        np.max(fwd[v] - fwd[u], axis=1),
        np.max(rev[u] - rev[v], axis=1),
    )
    lower = np.maximum(lower, 0.0)
    upper = np.min(rev[u] + fwd[v], axis=1)
    return lower.astype(np.float64), upper.astype(np.float64)


def metrics(pred, y, reverse, short_threshold):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    rel = np.abs(pred - y) / y
    alpha = np.abs(y - reverse) / ((y + reverse) / 2.0)
    short = y <= short_threshold
    asym = alpha >= 0.2
    return {
        'mre_percent': float(100.0 * rel.mean()),
        'mae': float(np.abs(pred - y).mean()),
        'short_mre_percent': float(100.0 * rel[short].mean()) if np.any(short) else None,
        'high_asymmetry_mre_percent': float(100.0 * rel[asym].mean()) if np.any(asym) else None,
        'high_asymmetry_count': int(asym.sum()),
    }


def build_common(workload: Path, matrix: Path):
    z = np.load(workload)
    train = z['train'].copy()
    val = z['validation'].copy()
    coords = z['coordinates'].astype(np.float32)
    n = len(coords)
    D = np.load(matrix, mmap_mode='r')
    if D.shape != (n, n):
        raise AssertionError((D.shape, n))
    if not np.isfinite(D).all():
        raise RuntimeError('distance matrix contains nonfinite entries')
    train_nodes = np.unique(train[:, :2].astype(np.int64))
    rng = np.random.default_rng(LANDMARK_SEED)
    landmarks = np.sort(rng.choice(train_nodes, size=32, replace=False))
    node = np.concatenate((np.asarray(D[landmarks, :]).T, np.asarray(D[:, landmarks])), axis=1).astype(np.float32)
    short = float(np.quantile(train[:, 2], 0.25))
    return z, train, val, coords, D, node, landmarks, short


def run_alt32(train, test, D, node, short):
    pred, upper = directed_bounds(node, test[:, :2])
    y = test[:, 2].astype(np.float64)
    reverse = np.asarray(D[test[:, 1].astype(np.int64), test[:, 0].astype(np.int64)], dtype=np.float64)
    if np.any(pred > y + 1e-5) or np.any(upper < y - 1e-5):
        raise AssertionError('directed landmark certificate violation in ALT32 audit')
    return metrics(pred, y, reverse, short)


class LandmarkNN(nn.Module):
    def __init__(self, d, max_distance):
        super().__init__()
        self.max_distance = float(max_distance)
        self.net = nn.Sequential(
            nn.Linear(d, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) * self.max_distance


def predict_nn(model, X, device):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 32768):
            out.append(model(torch.as_tensor(X[i:i + 32768], device=device)).cpu().numpy())
    return np.concatenate(out)


def run_catboost(train, val, test, coords, D, node, short, time_limit_s, out_model: Path):
    from catboost import CatBoostRegressor, Pool

    b, l1, _ = encode_numpy(node, coords, train[:, :2], False)
    Xtr = np.concatenate((b, l1), 1)
    ytr = train[:, 2].astype(np.float64)
    b, l1, _ = encode_numpy(node, coords, val[:, :2], False)
    Xv = np.concatenate((b, l1), 1)
    yv = val[:, 2].astype(np.float64)
    train_pool = Pool(Xtr, ytr)
    val_pool = Pool(Xv, yv)
    current = None
    total = 0
    history = []
    started = time.perf_counter()
    while time.perf_counter() - started < time_limit_s:
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            random_seed=1234,
            loss_function='RMSE',
            eval_metric='MAPE',
            task_type='CPU',
            thread_count=4,
            verbose=False,
        )
        model.fit(train_pool, init_model=current, eval_set=val_pool, verbose=False)
        current = model
        total += 500
        pv = current.predict(Xv)
        vm = float(100 * np.mean(np.abs(pv - yv) / yv))
        elapsed = time.perf_counter() - started
        history.append({'trees': total, 'elapsed_seconds': elapsed, 'validation_mre_percent': vm})
        print('R4N_DIR_CATBOOST_PROGRESS', total, elapsed, vm, flush=True)
        if elapsed >= time_limit_s:
            break
    if current is None:
        raise RuntimeError('CatBoost produced no model')
    b, l1, _ = encode_numpy(node, coords, test[:, :2], False)
    Xt = np.concatenate((b, l1), 1)
    y = test[:, 2].astype(np.float64)
    reverse = np.asarray(D[test[:, 1].astype(np.int64), test[:, 0].astype(np.int64)], dtype=np.float64)
    pred = current.predict(Xt)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    current.save_model(out_model)
    return {
        'feature_dim': int(Xtr.shape[1]),
        'trees': total,
        'train_seconds': time.perf_counter() - started,
        'history': history,
        'test': metrics(pred, y, reverse, short),
        'model_path': str(out_model),
    }


def run_landmarknn(train, val, test, coords, D, node, short, time_limit_s, seeds, out_dir: Path):
    b, _, l2 = encode_numpy(node, coords, train[:, :2], True)
    Xtr = np.concatenate((b, l2), 1)
    ytr = train[:, 2].astype(np.float32)
    b, _, l2 = encode_numpy(node, coords, val[:, :2], True)
    Xv = np.concatenate((b, l2), 1)
    yv = val[:, 2].astype(np.float32)
    maxd = float(np.max(ytr))
    device = 'cuda'
    runs = []
    states = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LandmarkNN(Xtr.shape[1], maxd).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        rng = np.random.default_rng(seed + 20260914)
        best = np.inf
        best_state = None
        best_step = 0
        history = []
        started = time.perf_counter()
        step = 0
        while time.perf_counter() - started < time_limit_s:
            step += 1
            ix = rng.integers(0, len(Xtr), size=BATCH)
            x = torch.as_tensor(Xtr[ix], device=device)
            y = torch.as_tensor(ytr[ix], device=device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred / maxd, y / maxd)
            loss.backward()
            opt.step()
            if step % 100 == 0:
                pv = predict_nn(model, Xv, device)
                vm = float(100 * np.mean(np.abs(pv - yv) / yv))
                elapsed = time.perf_counter() - started
                history.append({'step': step, 'elapsed_seconds': elapsed, 'validation_mre_percent': vm})
                if vm < best:
                    best = vm
                    best_step = step
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                print('R4N_DIR_LANDMARKNN_PROGRESS', seed, step, elapsed, best, flush=True)
        if best_state is None:
            raise RuntimeError(f'no validation checkpoint for seed {seed}')
        runs.append({
            'seed': int(seed),
            'best_step': best_step,
            'best_validation_mre_percent': best,
            'train_seconds': time.perf_counter() - started,
            'history': history,
        })
        states.append(best_state)

    # Test labels are used only after all requested seed states are frozen.
    b, _, l2 = encode_numpy(node, coords, test[:, :2], True)
    Xt = np.concatenate((b, l2), 1)
    y = test[:, 2].astype(np.float64)
    reverse = np.asarray(D[test[:, 1].astype(np.int64), test[:, 0].astype(np.int64)], dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    vals = []
    for row, state in zip(runs, states):
        model = LandmarkNN(Xtr.shape[1], maxd).to(device)
        model.load_state_dict(state)
        pred = predict_nn(model, Xt, device)
        met = metrics(pred, y, reverse, short)
        row['test'] = met
        vals.append(met['mre_percent'])
        ckpt = out_dir / f"seed{row['seed']}.pt"
        torch.save({'state_dict': state, 'input_dim': Xtr.shape[1], 'max_distance': maxd}, ckpt)
        row['checkpoint'] = str(ckpt)
    return {
        'feature_dim': int(Xtr.shape[1]),
        'runs': runs,
        'test_mre_mean': float(np.mean(vals)),
        'test_mre_sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--workload', type=Path, required=True)
    ap.add_argument('--matrix', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--models', nargs='+', choices=['alt32', 'catboost', 'landmarknn'], default=['alt32', 'catboost', 'landmarknn'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS_FINAL))
    ap.add_argument('--time-limit-seconds', type=float, default=300.0)
    args = ap.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = tuple(args.seeds)
    if not seeds:
        raise ValueError('at least one seed required')
    z, train, val, coords, D, node, landmarks, short = build_common(args.workload, args.matrix)
    # Load test only after all training inputs/landmarks are fixed; individual model functions
    # still defer test evaluation until their model state is frozen.
    test = np.load(args.workload)['test'].copy()

    rec = {
        'status': 'completed',
        'dataset': args.dataset,
        'classification': 'final directed-feature baseline evaluation' if seeds == SEEDS_FINAL and args.time_limit_seconds == 300.0 else 'bounded directed-feature baseline screening; not final stochastic-baseline evidence',
        'workload_sha256': sha256(args.workload),
        'matrix_sha256': sha256(args.matrix),
        'node_count': int(len(coords)),
        'train_rows': int(len(train)),
        'validation_rows': int(len(val)),
        'test_rows': int(len(test)),
        'landmark_seed': LANDMARK_SEED,
        'landmarks': landmarks.tolist(),
        'index_scalars_per_node': 64,
        'index_bytes_per_node_float32': 256,
        'requested_seeds': list(seeds),
        'time_limit_seconds_per_model_seed': float(args.time_limit_seconds),
        'models': {},
    }
    if 'alt32' in args.models:
        rec['models']['ALT32'] = {'test': run_alt32(train, test, D, node, short)}
    if 'catboost' in args.models:
        rec['models']['Dir-CatBoost'] = run_catboost(
            train, val, test, coords, D, node, short, args.time_limit_seconds,
            args.model_dir / 'dir_catboost.cbm',
        )
    if 'landmarknn' in args.models:
        rec['models']['Dir-LandmarkNN'] = run_landmarknn(
            train, val, test, coords, D, node, short, args.time_limit_seconds, seeds,
            args.model_dir / 'dir_landmarknn',
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2) + '\n')
    print('R4N_MATRIX_STRONG_BASELINES_COMPLETE', json.dumps({
        'dataset': args.dataset,
        'classification': rec['classification'],
        'results': {
            name: (body.get('test', {}).get('mre_percent') if 'test' in body else body.get('test_mre_mean'))
            for name, body in rec['models'].items()
        },
    }), flush=True)


if __name__ == '__main__':
    main()

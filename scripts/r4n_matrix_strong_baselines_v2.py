"""Protocol-preserving directed strong baselines on frozen matrix-backed workloads.

Generalizes the already-authorized Jinan R4F directed CatBoost/LandmarkNN adaptation
to Shenzhen/Chengdu-style frozen workloads. It does not introduce a new model family.
Test split is loaded only after each learned model (or all requested seeds of that
model) has been frozen by validation.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEEDS_FINAL = (42, 99, 1234)
LANDMARK_SEED = 20260914
BATCH = 16384
LR = 1e-3


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(a):
    mu = a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    return mu, np.where(sd < 1e-8, 1.0, sd)


def encode(node, coords, pairs, normalized=False):
    pairs = np.asarray(pairs, dtype=np.int64)
    u, v = pairs.T
    land = np.asarray(node, dtype=np.float32)
    xy = np.asarray(coords, dtype=np.float32)
    if normalized:
        lm, ls = stats(land); cm, cs = stats(xy)
        land = (land - lm) / ls; xy = (xy - cm) / cs
    a, b = land[u], land[v]
    ca, cb = xy[u], xy[v]
    dot = np.sum(a * b, axis=1)
    den = np.maximum(np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-12)
    cos = (dot / den)[:, None].astype(np.float32)
    l1 = np.abs(ca - cb).sum(1, keepdims=True).astype(np.float32)
    l2 = np.sqrt(np.square(ca - cb).sum(1, keepdims=True)).astype(np.float32)
    return np.concatenate((a, b, ca, cb, cos), axis=1).astype(np.float32), l1, l2


def directed_bounds(node, pairs):
    pairs = np.asarray(pairs, dtype=np.int64); u, v = pairs.T
    k = node.shape[1] // 2
    fwd, rev = node[:, :k], node[:, k:]
    lo = np.maximum(np.max(fwd[v] - fwd[u], axis=1), np.max(rev[u] - rev[v], axis=1))
    lo = np.maximum(lo, 0.0)
    hi = np.min(rev[u] + fwd[v], axis=1)
    return lo.astype(np.float64), hi.astype(np.float64)


def metric(pred, y, reverse, short_threshold):
    pred = np.asarray(pred, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    rel = np.abs(pred - y) / y
    alpha = np.abs(y - reverse) / ((y + reverse) / 2.0)
    short = y <= short_threshold; asym = alpha >= 0.2
    return {
        'mre_percent': float(100 * rel.mean()),
        'mae': float(np.abs(pred - y).mean()),
        'short_mre_percent': float(100 * rel[short].mean()) if np.any(short) else None,
        'high_asymmetry_mre_percent': float(100 * rel[asym].mean()) if np.any(asym) else None,
        'high_asymmetry_count': int(asym.sum()),
    }


def load_train_val(workload: Path, matrix: Path):
    z = np.load(workload)
    train = z['train'].copy(); val = z['validation'].copy(); coords = z['coordinates'].astype(np.float32)
    n = len(coords); D = np.load(matrix, mmap_mode='r')
    if D.shape != (n, n) or not np.isfinite(D).all():
        raise RuntimeError(f'bad distance matrix {D.shape} for n={n}')
    train_nodes = np.unique(train[:, :2].astype(np.int64))
    rng = np.random.default_rng(LANDMARK_SEED)
    landmarks = np.sort(rng.choice(train_nodes, size=32, replace=False))
    node = np.concatenate((np.asarray(D[landmarks, :]).T, np.asarray(D[:, landmarks])), axis=1).astype(np.float32)
    short = float(np.quantile(train[:, 2], .25))
    return train, val, coords, D, node, landmarks, short


def load_test(workload: Path):
    return np.load(workload)['test'].copy()


class LandmarkNN(nn.Module):
    def __init__(self, d, max_distance):
        super().__init__(); self.max_distance = float(max_distance)
        self.net = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1) * self.max_distance


def predict_nn(model, X, device='cuda'):
    out = []; model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 32768):
            out.append(model(torch.as_tensor(X[i:i+32768], device=device)).cpu().numpy())
    return np.concatenate(out)


def eval_alt32(workload, D, node, short):
    test = load_test(workload)
    lo, hi = directed_bounds(node, test[:, :2]); y = test[:, 2].astype(np.float64)
    if np.any(lo > y + 1e-5) or np.any(hi < y - 1e-5):
        raise AssertionError('directed landmark certificate violation')
    rev = np.asarray(D[test[:,1].astype(np.int64), test[:,0].astype(np.int64)], dtype=np.float64)
    return {'test': metric(lo, y, rev, short)}


def train_catboost(train, val, coords, node, limit_s):
    from catboost import CatBoostRegressor, Pool
    b, l1, _ = encode(node, coords, train[:, :2], False); Xtr = np.concatenate((b, l1), 1); ytr = train[:,2].astype(np.float64)
    b, l1, _ = encode(node, coords, val[:, :2], False); Xv = np.concatenate((b, l1), 1); yv = val[:,2].astype(np.float64)
    current = None; total = 0; history = []; started = time.perf_counter()
    while time.perf_counter() - started < limit_s:
        m = CatBoostRegressor(iterations=500, learning_rate=.1, random_seed=1234, loss_function='RMSE', eval_metric='MAPE', task_type='CPU', thread_count=4, verbose=False)
        m.fit(Pool(Xtr, ytr), init_model=current, eval_set=Pool(Xv, yv), verbose=False)
        current = m; total += 500
        pv = current.predict(Xv); vm = float(100*np.mean(np.abs(pv-yv)/yv)); elapsed = time.perf_counter()-started
        history.append({'trees':total,'elapsed_seconds':elapsed,'validation_mre_percent':vm})
        print('R4N_DIR_CATBOOST_PROGRESS', total, elapsed, vm, flush=True)
        if elapsed >= limit_s: break
    if current is None: raise RuntimeError('no CatBoost model')
    return current, {'feature_dim':int(Xtr.shape[1]),'trees':total,'train_seconds':time.perf_counter()-started,'history':history}


def eval_catboost(model, rec, workload, coords, D, node, short, model_path):
    test = load_test(workload)
    b, l1, _ = encode(node, coords, test[:, :2], False); Xt = np.concatenate((b, l1), 1); y = test[:,2].astype(np.float64)
    rev = np.asarray(D[test[:,1].astype(np.int64), test[:,0].astype(np.int64)], dtype=np.float64)
    pred = model.predict(Xt); model_path.parent.mkdir(parents=True, exist_ok=True); model.save_model(model_path)
    rec = dict(rec); rec['test'] = metric(pred, y, rev, short); rec['model_path'] = str(model_path)
    return rec


def train_landmarknn(train, val, coords, node, limit_s, seeds):
    b, _, l2 = encode(node, coords, train[:, :2], True); Xtr = np.concatenate((b, l2), 1); ytr = train[:,2].astype(np.float32)
    b, _, l2 = encode(node, coords, val[:, :2], True); Xv = np.concatenate((b, l2), 1); yv = val[:,2].astype(np.float32)
    maxd = float(np.max(ytr)); runs=[]; states=[]
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        m=LandmarkNN(Xtr.shape[1],maxd).cuda(); opt=torch.optim.Adam(m.parameters(),lr=LR); rng=np.random.default_rng(seed+20260914)
        best=np.inf; best_state=None; best_step=0; history=[]; started=time.perf_counter(); step=0
        while time.perf_counter()-started < limit_s:
            step += 1; ix=rng.integers(0,len(Xtr),size=BATCH)
            x=torch.as_tensor(Xtr[ix],device='cuda'); y=torch.as_tensor(ytr[ix],device='cuda')
            opt.zero_grad(set_to_none=True); pred=m(x); loss=F.mse_loss(pred/maxd,y/maxd); loss.backward(); opt.step()
            if step%100==0:
                pv=predict_nn(m,Xv); vm=float(100*np.mean(np.abs(pv-yv)/yv)); elapsed=time.perf_counter()-started
                history.append({'step':step,'elapsed_seconds':elapsed,'validation_mre_percent':vm})
                if vm<best: best=vm; best_step=step; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
                print('R4N_DIR_LANDMARKNN_PROGRESS',seed,step,elapsed,best,flush=True)
        if best_state is None: raise RuntimeError(f'no validation checkpoint seed={seed}')
        runs.append({'seed':int(seed),'best_step':best_step,'best_validation_mre_percent':best,'train_seconds':time.perf_counter()-started,'history':history}); states.append(best_state)
    return Xtr.shape[1], maxd, runs, states


def eval_landmarknn(input_dim, maxd, runs, states, workload, coords, D, node, short, out_dir):
    test=load_test(workload); b,_,l2=encode(node,coords,test[:,:2],True); Xt=np.concatenate((b,l2),1); y=test[:,2].astype(np.float64)
    rev=np.asarray(D[test[:,1].astype(np.int64),test[:,0].astype(np.int64)],dtype=np.float64); vals=[]; out_dir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=LandmarkNN(input_dim,maxd).cuda(); m.load_state_dict(state); pred=predict_nn(m,Xt); row['test']=metric(pred,y,rev,short); vals.append(row['test']['mre_percent'])
        p=out_dir/f"seed{row['seed']}.pt"; torch.save({'state_dict':state,'input_dim':input_dim,'max_distance':maxd},p); row['checkpoint']=str(p)
    return {'feature_dim':int(input_dim),'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--workload',type=Path,required=True); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--models',nargs='+',choices=['alt32','catboost','landmarknn'],default=['alt32','catboost','landmarknn']); ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS_FINAL)); ap.add_argument('--time-limit-seconds',type=float,default=300.0); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    seeds=tuple(args.seeds); train,val,coords,D,node,landmarks,short=load_train_val(args.workload,args.matrix)
    final=(seeds==SEEDS_FINAL and float(args.time_limit_seconds)==300.0)
    rec={'status':'completed','dataset':args.dataset,'classification':'final directed-feature baseline evaluation' if final else 'bounded directed-feature baseline screening; not final stochastic-baseline evidence','workload_sha256':sha256(args.workload),'matrix_sha256':sha256(args.matrix),'node_count':int(len(coords)),'train_rows':int(len(train)),'validation_rows':int(len(val)),'landmark_seed':LANDMARK_SEED,'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'requested_seeds':list(seeds),'time_limit_seconds_per_model_seed':float(args.time_limit_seconds),'models':{}}
    if 'alt32' in args.models: rec['models']['ALT32']=eval_alt32(args.workload,D,node,short)
    if 'catboost' in args.models:
        m,r=train_catboost(train,val,coords,node,args.time_limit_seconds); rec['models']['Dir-CatBoost']=eval_catboost(m,r,args.workload,coords,D,node,short,args.model_dir/'dir_catboost.cbm')
    if 'landmarknn' in args.models:
        d,maxd,runs,states=train_landmarknn(train,val,coords,node,args.time_limit_seconds,seeds); rec['models']['Dir-LandmarkNN']=eval_landmarknn(d,maxd,runs,states,args.workload,coords,D,node,short,args.model_dir/'dir_landmarknn')
    rec['test_rows']=int(len(load_test(args.workload)))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4N_MATRIX_STRONG_BASELINES_COMPLETE',json.dumps({'dataset':args.dataset,'classification':rec['classification'],'summary':{k:(v.get('test',{}).get('mre_percent') if 'test' in v else v.get('test_mre_mean')) for k,v in rec['models'].items()}}),flush=True)

if __name__=='__main__': main()

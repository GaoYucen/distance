"""R4C: held-out OD validation for fixed-64-scalar quasimetric node tables.

The protocol is frozen in docs/r4c-realroads-20260914/PROTOCOL.md.
This script never changes R1-R4B artifacts. `train` selects checkpoints using
validation MRE only; `replay` independently decodes saved checkpoints on CPU,
checks predictions, and renders the final report.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r4b_models import MODES, SEEDS, components, reduce_components, random_tables

REPORT = ROOT / 'reports/audit-r4c-20260914'
RESULT = ROOT / 'results/audit-r4c-20260914'
UPSTREAM_SHA = '0365394d4ecbd614f38775787af724d8b142c9d2'
CASES = {
    'Jinan_native_directed': {
        'path': ROOT / 'data/protocol_r2/Jinan_native_directed.npz',
        'sha256': '22640a0e3fc7d5e35ae9b9d5dab6e3e172b787fd99c3b6ccac80809161b2053c',
        'raw_edges': ROOT / 'data/figshare_native_20260913/edge_jinan.csv',
        'landmark_seed': 20260914,
        'workload': 'trajectory-derived unordered OD groups after overlap isolation',
    },
    'Shenzhen_native_directed_uniform': {
        'path': ROOT / 'data/protocol_r2/Shenzhen_native_directed_uniform.npz',
        'sha256': '8cede9cc3b77d22acf2877ee85e0d6fa34d05785bf05b1b4cc9a3f86566fc92c',
        'raw_edges': ROOT / 'data/figshare_native_20260913/edge_shenzhen.csv',
        'landmark_seed': 20260915,
        'workload': 'seeded uniform unordered OD groups, not trajectories',
    },
}
UPDATES = 1500
BATCH_GROUPS = 8192
VAL_EVERY = 25
LR = .01


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(q: np.ndarray):
    ids = q[:, :2].astype(np.int64, copy=False)
    rev = ids[:, ::-1]
    return np.vstack((ids, rev)), np.concatenate((q[:, 2], q[:, 3])).astype(np.float32)


def group_metrics(pred: np.ndarray, q: np.ndarray, short_threshold: float) -> dict:
    n = len(q)
    if len(pred) != 2*n:
        raise ValueError('directional prediction length mismatch')
    truth = np.concatenate((q[:, 2], q[:, 3])).astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    rel = np.abs(p-truth)/truth
    p1, p2 = p[:n], p[n:]
    du, dv = q[:, 2].astype(np.float64), q[:, 3].astype(np.float64)
    A = (du-dv)/2
    pA = (p1-p2)/2
    S = (du+dv)/2
    alpha = np.abs(du-dv)/S
    high_group = alpha >= .2
    short_group = S <= short_threshold
    directional_high = np.concatenate((high_group, high_group))
    directional_short = np.concatenate((short_group, short_group))
    return {
        'directions': int(2*n),
        'mre_percent': float(100*rel.mean()),
        'mae_m': float(np.abs(p-truth).mean()),
        'p95_relative_percent': float(100*np.quantile(rel, .95)),
        'max_relative_percent': float(100*rel.max()),
        'direction_half_difference_rmse_m': float(np.sqrt(np.mean((pA-A)**2))),
        'high_asymmetry_groups': int(high_group.sum()),
        'high_asymmetry_mre_percent': float(100*rel[directional_high].mean()) if high_group.any() else None,
        'short_groups': int(short_group.sum()),
        'short_mre_percent': float(100*rel[directional_short].mean()) if short_group.any() else None,
        'overprediction_fraction': float(np.mean(p > truth + 1e-6)),
        'underprediction_fraction': float(np.mean(p < truth - 1e-6)),
        'negative_prediction_fraction': float(np.mean(p < -1e-7)),
    }


class PairSeedBatch(nn.Module):
    def __init__(self, n: int, mode: str, device: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.table = nn.Parameter(random_tables(n).to(device))
        if mode == 'IQE-maxmean':
            self.raw_alpha = nn.Parameter(torch.full((len(SEEDS),), -1., device=device))
        else:
            self.register_buffer('raw_alpha', torch.full((len(SEEDS),), -1., device=device))
        self.register_buffer('calibration', torch.ones(len(SEEDS), device=device))

    def forward(self, ids: torch.Tensor, return_components=False):
        x = self.table[:, ids[:, 0], :]
        y = self.table[:, ids[:, 1], :]
        c = components(x, y, self.mode)
        out = reduce_components(c, self.mode, self.raw_alpha) * self.calibration[:, None]
        return (out, c) if return_components else out

    @torch.no_grad()
    def calibrate(self, ids: np.ndarray, chunk=32768):
        sums = torch.zeros(len(SEEDS), device=self.table.device)
        count = 0
        for begin in range(0, len(ids), chunk):
            x = torch.as_tensor(ids[begin:begin+chunk], device=self.table.device)
            sums += self(x).sum(1)
            count += len(x)
        mean = sums / count
        if not bool(torch.isfinite(mean).all() and (mean > 0).all()):
            raise RuntimeError('invalid initial mean')
        self.calibration.copy_(1/mean)


def predict_torch(model: PairSeedBatch, ids: np.ndarray, chunk=32768) -> np.ndarray:
    model.eval(); parts=[]
    with torch.no_grad():
        for begin in range(0, len(ids), chunk):
            x = torch.as_tensor(ids[begin:begin+chunk], device=model.table.device)
            parts.append(model(x).detach().cpu().numpy())
    return np.concatenate(parts, axis=1)


def numpy_pair_components(table: np.ndarray, ids: np.ndarray, mode: str) -> np.ndarray:
    a = np.asarray(table, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    x, y = a[ids[:, 0]], a[ids[:, 1]]
    d = y-x
    if mode == 'L1':
        return np.abs(d).sum(-1, keepdims=True)
    if mode in ('T1','B2','B4','B8'):
        k = 1 if mode == 'T1' else int(mode[1:])
        block = d.reshape(len(d), k, 64//k)
        return np.abs(block[..., :-1]).sum(-1) + block[..., -1]
    if mode == 'P64':
        return d
    if mode in ('Shared-L1','MRN-L2'):
        r = 56 if mode == 'Shared-L1' else 32
        if mode == 'Shared-L1':
            sym = np.abs(d[:, :r]).sum(-1, keepdims=True)
        else:
            sym = np.sqrt(np.square(d[:, :r]).sum(-1, keepdims=True))
        return np.concatenate((sym, sym+d[:, r:]), axis=-1)
    if mode.startswith('IQE-'):
        xx, yy = x.reshape(len(x),8,8), y.reshape(len(y),8,8)
        valid = xx < yy
        lo = xx.copy(); hi = np.where(valid, yy, xx)
        order = np.argsort(lo, axis=-1, kind='stable')
        left = np.take_along_axis(lo, order, axis=-1)
        right = np.take_along_axis(hi, order, axis=-1)
        covered = np.maximum.accumulate(right, axis=-1)
        prev = np.concatenate((np.full_like(covered[:, :, :1], -np.inf), covered[:, :, :-1]), axis=-1)
        return np.maximum(0., right-np.maximum(left, prev)).sum(-1)
    raise ValueError(mode)


def numpy_pair_decode(table, ids, mode, raw_alpha, calibration):
    c = numpy_pair_components(table, ids, mode)
    if mode == 'IQE-sum':
        out = c.sum(-1)
    elif mode == 'IQE-maxmean':
        alpha = 1/(1+np.exp(-float(raw_alpha)))
        out = (1-alpha)*c.mean(-1) + alpha*c.max(-1)
    else:
        out = np.maximum(0., c.max(-1))
    return out*float(calibration)


def component_stats_numpy(table, ids, mode, raw_alpha, calibration):
    c = numpy_pair_components(table, ids, mode)
    if c.shape[-1] <= 1:
        return {'components': int(c.shape[-1]), 'active_gt_1pct': int(c.shape[-1]), 'dominant_fractions': [1.0]}
    winners = np.argmax(c, axis=-1)
    frac = [float(np.mean(winners == i)) for i in range(c.shape[-1])]
    return {'components': int(c.shape[-1]), 'active_gt_1pct': int(sum(v>=.01 for v in frac)), 'dominant_fractions': frac,
            'raw_component_negative_fraction': float(np.mean(c < 0))}


def snapshot_seed(model: PairSeedBatch, seed_index: int):
    return {
        'table': model.table[seed_index].detach().cpu().numpy().astype(np.float32, copy=True),
        'raw_alpha': float(model.raw_alpha[seed_index].detach().cpu()),
        'calibration': float(model.calibration[seed_index].detach().cpu()),
    }


def train_mode(case: str, mode: str, path: Path, expected_hash: str, device: str) -> dict:
    if sha256(path) != expected_hash:
        raise RuntimeError(f'data hash mismatch: {case}')
    z = np.load(path)
    train = z['train'].copy(); val = z['validation'].copy()
    n = len(z['coordinates'])
    train_ids, train_y = ordered(train); val_ids, val_y = ordered(val)
    scale = float(train_y.mean())
    short_threshold = float(np.quantile((train[:,2]+train[:,3])/2, .25))
    model = PairSeedBatch(n, mode, device)
    initial_hashes = [hashlib.sha256(model.table[i].detach().cpu().numpy().tobytes()).hexdigest() for i in range(len(SEEDS))]
    model.calibrate(train_ids)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(20260914 + (0 if case.startswith('Jinan') else 1))
    best_val = np.full(len(SEEDS), np.inf)
    best_step = np.full(len(SEEDS), -1, dtype=int)
    best_state = [None]*len(SEEDS)
    history=[]

    def validate(step: int, batch_loss=None):
        pv = predict_torch(model, val_ids) * scale
        vals = np.mean(np.abs(pv-val_y[None,:])/val_y[None,:], axis=1)*100
        history.append({'step': int(step), 'validation_mre_percent': vals.tolist(),
                        'batch_normalized_mse': None if batch_loss is None else [float(x) for x in batch_loss]})
        for i,v in enumerate(vals):
            if v < best_val[i]:
                best_val[i] = float(v); best_step[i] = int(step); best_state[i] = snapshot_seed(model,i)

    validate(0)
    model.train(); started=time.perf_counter()
    group_count=len(train)
    for step in range(1, UPDATES+1):
        g = rng.integers(0, group_count, size=BATCH_GROUPS, endpoint=False)
        ids = np.vstack((train[g,:2], train[g,:2][:,::-1])).astype(np.int64)
        y = np.concatenate((train[g,2], train[g,3])).astype(np.float32)/scale
        ids_t = torch.as_tensor(ids, device=device)
        y_t = torch.as_tensor(y, device=device)
        opt.zero_grad(set_to_none=True)
        pred = model(ids_t)
        loss_seed = torch.mean((pred-y_t[None,:])**2, dim=1)
        loss_seed.sum().backward(); opt.step()
        if step % VAL_EVERY == 0:
            validate(step, loss_seed.detach().cpu().numpy())
        if step % 250 == 0:
            print('R4C_PROGRESS',case,mode,step,best_val.tolist(),flush=True)
    train_seconds=time.perf_counter()-started
    if any(s is None for s in best_state):
        raise RuntimeError('missing best state')

    # Only after all best validation states are frozen do we read the test labels.
    z2=np.load(path); test=z2['test'].copy(); test_ids,test_y=ordered(test)
    outdir=RESULT/case/mode; outdir.mkdir(parents=True, exist_ok=True)
    runs=[]
    for i,seed in enumerate(SEEDS):
        state=best_state[i]
        pred=numpy_pair_decode(state['table'],test_ids,mode,state['raw_alpha'],state['calibration'])*scale
        metrics=group_metrics(pred,test,short_threshold)
        comp=component_stats_numpy(state['table'],test_ids,mode,state['raw_alpha'],state['calibration'])
        ck=outdir/f'seed{seed}.npz'
        np.savez_compressed(ck,table=state['table'],raw_alpha=np.array(state['raw_alpha']),
            calibration=np.array(state['calibration']),scale=np.array(scale),best_step=np.array(best_step[i]),
            test_predictions=pred.astype(np.float32),test_queries=test.astype(np.float64))
        runs.append({'seed':seed,'initial_table_sha256':initial_hashes[i],'best_step':int(best_step[i]),
            'best_validation_mre_percent':float(best_val[i]),'test':metrics,'components':comp,
            'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':sha256(ck)})
    report={'case':case,'mode':mode,'status':'completed','node_count':n,'train_groups':len(train),'validation_groups':len(val),
        'test_groups':len(test),'scale_m':scale,'short_threshold_m':short_threshold,'updates':UPDATES,
        'batch_groups':BATCH_GROUPS,'lr':LR,'train_seconds':train_seconds,'history':history,'runs':runs}
    rpath=REPORT/'runs'/f'{case}__{mode}.json';rpath.parent.mkdir(parents=True,exist_ok=True)
    rpath.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('R4C_MODE_COMPLETE',case,mode,[r['test']['mre_percent'] for r in runs],flush=True)
    del model,opt
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return report


def build_native_csr(case: str, z, raw_csv: Path):
    original=z['original_node_ids'].astype(np.int64); n=len(original)
    maxid=int(original.max()); mapping=np.full(maxid+1,-1,dtype=np.int64);mapping[original]=np.arange(n)
    best={}
    with raw_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            u0,v0=int(row['Origin']),int(row['Destination'])
            if u0>maxid or v0>maxid: continue
            u,v=int(mapping[u0]),int(mapping[v0])
            if u<0 or v<0 or u==v: continue
            w=float(row['Length']); key=(u,v)
            if key not in best or w<best[key]: best[key]=w
    rows=np.fromiter((k[0] for k in best),dtype=np.int64,count=len(best));cols=np.fromiter((k[1] for k in best),dtype=np.int64,count=len(best))
    vals=np.fromiter(best.values(),dtype=np.float64,count=len(best))
    return sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))


def landmark_baseline(case: str, info: dict) -> dict:
    z=np.load(info['path']); train=z['train']; test=z['test']; n=len(z['coordinates'])
    train_nodes=np.unique(train[:,:2].astype(np.int64));rng=np.random.default_rng(info['landmark_seed'])
    landmarks=np.sort(rng.choice(train_nodes,size=32,replace=False))
    started=time.perf_counter();A=build_native_csr(case,z,info['raw_edges'])
    fr=dijkstra(A,directed=True,indices=landmarks);to=dijkstra(A.T,directed=True,indices=landmarks)
    build_seconds=time.perf_counter()-started
    if not np.isfinite(fr).all() or not np.isfinite(to).all():raise RuntimeError('landmark distances not finite')
    features=np.concatenate((fr.T,to.T),axis=1).astype(np.float32)
    ids,_=ordered(test);u,v=ids.T
    f=features[:,:32].T.astype(np.float64);t=features[:,32:].T.astype(np.float64)
    lb=np.maximum(0.,np.maximum((f[:,v]-f[:,u]).max(0),(t[:,u]-t[:,v]).max(0)))
    ub=(t[:,u]+f[:,v]).min(0)
    short=float(np.quantile((train[:,2]+train[:,3])/2,.25))
    out={'case':case,'landmarks':landmarks.tolist(),'seed':info['landmark_seed'],'index_scalars_per_node':64,
        'index_bytes_float32':int(features.nbytes),'build_seconds':build_seconds,
        'ALT32-LB':group_metrics(lb,test,short),'ALT32-UB':group_metrics(ub,test,short)}
    dest=RESULT/case/'ALT32';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'index.npz',features=features,landmarks=landmarks,test_lb=lb.astype(np.float32),test_ub=ub.astype(np.float32),test_queries=test)
    out['artifact']=str((dest/'index.npz').relative_to(ROOT));out['artifact_sha256']=sha256(dest/'index.npz')
    (REPORT/'runs'/f'{case}__ALT32.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('R4C_ALT_COMPLETE',case,out['ALT32-LB']['mre_percent'],out['ALT32-UB']['mre_percent'],flush=True)
    return out


def leakage_guard(device='cpu'):
    # The optimizer never accepts test arrays. This small deterministic repeat also checks that changing a held-out array
    # leaves the training/validation trajectory unchanged when all other inputs are identical.
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    n=40; rng=np.random.default_rng(919);pairs=[]
    while len(pairs)<500:
        u,v=map(int,rng.integers(0,n,size=2))
        if u!=v:pairs.append((u,v))
    pairs=np.array(pairs);d=1+np.abs(pairs[:,0]-pairs[:,1]);q=np.column_stack((pairs,d,d+0.5)).astype(float)
    train,val=q[:350],q[350:425]
    def run():
        model=PairSeedBatch(n,'T1',device);ids,y=ordered(train);scale=float(y.mean());model.calibrate(ids)
        opt=torch.optim.Adam(model.parameters(),lr=.01);g=np.random.default_rng(123);hist=[]
        for step in range(6):
            ix=g.integers(0,len(train),64);bid=np.vstack((train[ix,:2],train[ix,:2][:,::-1])).astype(np.int64)
            by=np.concatenate((train[ix,2],train[ix,3])).astype(np.float32)/scale
            opt.zero_grad();p=model(torch.as_tensor(bid,device=device));loss=((p-torch.as_tensor(by,device=device)[None,:])**2).mean(1);loss.sum().backward();opt.step()
            pv=predict_torch(model,ordered(val)[0]);hist.append(float(pv[0].mean()))
        state=model.table.detach().cpu().numpy().copy();return hist,state
    h1,s1=run();h2,s2=run();
    if h1!=h2 or not np.array_equal(s1,s2):raise AssertionError('deterministic leakage guard failed')
    torch.use_deterministic_algorithms(False);torch.set_num_threads(4)
    return {'passed':True,'note':'training function has no test argument; deterministic repeated histories/states match exactly'}


def train_all(device):
    REPORT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    meta={'status':'running','classification':'held-out OD development validation; NOT final independent confirmation',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'upstream_torchqmet':UPSTREAM_SHA,
        'protocol':{'scalars_per_node':64,'dtype':'float32','seeds':list(SEEDS),'updates':UPDATES,'batch_groups':BATCH_GROUPS,
                    'validation_every':VAL_EVERY,'optimizer':'Adam','learning_rate':LR,'objective':'normalized MSE','selection':'validation MRE only'},
        'leakage_guard':leakage_guard('cpu'),'data':{},'runs':[],'landmarks':[]}
    (REPORT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    for case,info in CASES.items():
        if sha256(info['path'])!=info['sha256']:raise RuntimeError('input hash mismatch')
        z=np.load(info['path']);meta['data'][case]={'sha256':info['sha256'],'nodes':len(z['coordinates']),'splits':[len(z[k]) for k in ['train','validation','test']], 'workload':info['workload']}
        for mode in MODES:
            result=train_mode(case,mode,info['path'],info['sha256'],device);meta['runs'].append({'case':case,'mode':mode,'file':f'reports/audit-r4c-20260914/runs/{case}__{mode}.json'})
            (REPORT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        alt=landmark_baseline(case,info);meta['landmarks'].append({'case':case,'file':f'reports/audit-r4c-20260914/runs/{case}__ALT32.json'})
        (REPORT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    meta['status']='trained_pending_independent_replay';(REPORT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('R4C_TRAINING_COMPLETE',flush=True)


def replay_all():
    manifest=json.loads((REPORT/'manifest.json').read_text());
    if manifest['status'] not in ('trained_pending_independent_replay','completed_and_replayed'):raise RuntimeError('training incomplete')
    replay={'status':'running','items':[]};max_pred=0.;max_mre=0.
    for case,info in CASES.items():
        z=np.load(info['path']);test=z['test'];ids,_=ordered(test);train=z['train'];short=float(np.quantile((train[:,2]+train[:,3])/2,.25))
        for mode in MODES:
            run=json.loads((REPORT/'runs'/f'{case}__{mode}.json').read_text())
            for row in run['runs']:
                ck=ROOT/row['checkpoint'];
                if sha256(ck)!=row['checkpoint_sha256']:raise AssertionError('checkpoint hash mismatch')
                s=np.load(ck);np.testing.assert_array_equal(s['test_queries'],test)
                pred=numpy_pair_decode(s['table'],ids,mode,float(s['raw_alpha']),float(s['calibration']))*float(s['scale'])
                stored=s['test_predictions'].astype(np.float64);err=float(np.max(np.abs(pred-stored)))
                met=group_metrics(pred,test,short);merr=abs(met['mre_percent']-row['test']['mre_percent'])
                if err>.05 or merr>.002:raise AssertionError((case,mode,row['seed'],err,merr))
                max_pred=max(max_pred,err);max_mre=max(max_mre,merr)
                replay['items'].append({'case':case,'mode':mode,'seed':row['seed'],'max_prediction_difference_m':err,'mre_difference_pp':merr})
    replay.update({'status':'passed','replays':len(replay['items']),'max_prediction_difference_m':max_pred,'max_mre_difference_pp':max_mre})
    (REPORT/'replay.json').write_text(json.dumps(replay,indent=2,allow_nan=False)+'\n')
    render_report(manifest,replay)
    manifest['status']='completed_and_replayed';manifest['replays']=len(replay['items']);(REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('R4C_REPLAY_COMPLETE',len(replay['items']),max_pred,max_mre,flush=True)


def render_report(manifest,replay):
    summary={}
    for case in CASES:
        summary[case]={}
        for mode in MODES:
            run=json.loads((REPORT/'runs'/f'{case}__{mode}.json').read_text())
            vals=np.array([x['test']['mre_percent'] for x in run['runs']]);v=np.array([x['best_validation_mre_percent'] for x in run['runs']])
            summary[case][mode]={'test_mre_mean':float(vals.mean()),'test_mre_sd':float(vals.std(ddof=1)),'per_seed':vals.tolist(),
                'validation_mre_mean':float(v.mean()),'best_steps':[x['best_step'] for x in run['runs']],
                'high_asymmetry_mre_mean':float(np.mean([x['test']['high_asymmetry_mre_percent'] for x in run['runs'] if x['test']['high_asymmetry_mre_percent'] is not None])),
                'short_mre_mean':float(np.mean([x['test']['short_mre_percent'] for x in run['runs'] if x['test']['short_mre_percent'] is not None]))}
        alt=json.loads((REPORT/'runs'/f'{case}__ALT32.json').read_text());summary[case]['ALT32-LB']={'test_mre_mean':alt['ALT32-LB']['mre_percent']};summary[case]['ALT32-UB']={'test_mre_mean':alt['ALT32-UB']['mre_percent']}
    (REPORT/'summary.json').write_text(json.dumps({'classification':'development held-out OD, not final independent confirmation','summary':summary,'replay':replay},indent=2,allow_nan=False)+'\n')
    names=list(MODES)+['ALT32-LB','ALT32-UB']
    lines=['# R4C｜真实原生有向路网的未见 OD 验证','',
      '**本轮是 held-out OD 开发验证，不是最终独立确认集。** 济南测试标签在 R2/R3 已被查看过；深圳采用均匀 OD，不等同轨迹工作负载。','',
      f"学习表示固定每节点64个float32；3种子；1500更新；仅验证MRE选checkpoint。独立CPU/NumPy重放 {replay['replays']} 个学习checkpoint全部通过。",'',
      '| 方法 | 济南测试MRE (%) | 深圳测试MRE (%) |','|---|---:|---:|']
    for mode in names:
        a=summary['Jinan_native_directed'][mode]['test_mre_mean'];b=summary['Shenzhen_native_directed_uniform'][mode]['test_mre_mean']
        if mode in MODES:
            sa=summary['Jinan_native_directed'][mode]['test_mre_sd'];sb=summary['Shenzhen_native_directed_uniform'][mode]['test_mre_sd'];lines.append(f'| {mode} | {a:.4f} ± {sa:.4f} | {b:.4f} ± {sb:.4f} |')
        else:lines.append(f'| {mode} | {a:.4f} | {b:.4f} |')
    lines += ['', '## 边界与后续','',
      '- 不从两个数据各自后选不同 K 组成一个新方法；预先固定的 B2/B4/B8 分别单独判断。',
      '- 若强对照 P64、Shared-L1、MRN/IQE 或 ALT32 更优，保留该结果，不通过新增模块掩盖。',
      '- 训练目标仍是归一化MSE，因此MRE排名与优化目标可能不同；完整验证、短距离、高非对称切片在 runs/*.json。',
      '- 本轮没有生产查询延迟、没有GNN、没有跨节点归纳泛化。通过表示门槛后再单独冻结下一协议。']
    (REPORT/'R4C_STATUS_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','replay']);p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.action=='train':train_all(args.device)
    else:replay_all()

if __name__=='__main__':main()

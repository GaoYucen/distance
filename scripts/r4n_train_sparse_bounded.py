"""Run the already-frozen R4M architecture from a precomputed 64-scalar directed-landmark table.

This is data-path generalization only for large sparse graphs where an all-pairs matrix is infeasible.
Architecture, loss, optimizer, seeds, batch, landmark seed/index budget, validation selection, and 300 s/seed
are identical to r4n_train_matrix_bounded.py.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import scripts.r4m_bounded_landmarknn as r4m
import scripts.r4e_directed_landmark_residual as r4e

SEEDS=(42,99,1234); BATCH=16384; LR=1e-3; VAL_EVERY=100; TIME_LIMIT_S=300.0; LANDMARK_SEED=20260914

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--workload',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    z=np.load(args.workload); idx=np.load(args.index)
    train=z['train'].copy(); val=z['validation'].copy(); coords=z['coordinates'].astype(np.float32); n=len(coords)
    node=idx['features'].astype(np.float32); landmarks=idx['landmarks'].astype(np.int64)
    if node.shape!=(n,64) or landmarks.shape!=(32,): raise RuntimeError((node.shape,landmarks.shape,n))
    train_nodes=np.unique(train[:,:2].astype(np.int64)); rng=np.random.default_rng(LANDMARK_SEED); expected=np.sort(rng.choice(train_nodes,size=32,replace=False))
    if not np.array_equal(landmarks,expected): raise RuntimeError('landmark selection differs from frozen R4M protocol')
    scale=float(train[:,2].mean()); short=float(np.quantile(train[:,2],.25))
    Xtr,Ltr,Utr,mu,sd=r4m.build_features(node,coords,train[:,:2],scale); Xv,Lv,Uv,_,_=r4m.build_features(node,coords,val[:,:2],scale,mu,sd)
    ytr=train[:,2].astype(np.float64); yv=val[:,2].astype(np.float64); tol=1e-6
    if np.any(Ltr>ytr+tol) or np.any(Utr<ytr-tol) or np.any(Lv>yv+tol) or np.any(Uv<yv-tol): raise AssertionError('certified bounds violated')
    G=Utr-Ltr; mask=G>1e-12; astar=np.clip((ytr[mask]-Ltr[mask])/G[mask],0,1); weights=G[mask]/ytr[mask]; alpha0=r4m.weighted_median(astar,weights)
    Xvt=torch.as_tensor(Xv,device='cuda'); Lvt=torch.as_tensor(Lv,dtype=torch.float32,device='cuda'); Gvt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda'); yvt=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
    runs=[]; states=[]
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed); rr=np.random.default_rng(seed+20260914)
        m=r4m.BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5); best=np.inf; best_step=0; best_state=None; started=time.perf_counter(); step=0
        while time.perf_counter()-started<TIME_LIMIT_S:
            step+=1; ix=rr.integers(0,len(Xtr),size=BATCH); x=torch.as_tensor(Xtr[ix],device='cuda'); l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device='cuda'); g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device='cuda'); y=torch.as_tensor(ytr[ix],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True); p=l+m(x)*g; re=(p-y)/y; loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02); loss.backward(); opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad(): pv=Lvt+m(Xvt)*Gvt; vm=float(torch.mean(torch.abs(pv-yvt)/yvt)*100)
                m.train()
                if vm<best: best=vm; best_step=step; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%1000==0: print('R4N_SPARSE_PROGRESS',args.dataset,seed,step,best,best_step,flush=True)
        if best_state is None: raise RuntimeError('no validation checkpoint')
        runs.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'train_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in m.parameters())}); states.append(best_state)
    test=np.load(args.workload)['test'].copy(); reverse=np.load(args.workload)['test_reverse_distances'].astype(np.float64)
    Xt,Lt,Ut,_,_=r4m.build_features(node,coords,test[:,:2],scale,mu,sd); yt=test[:,2].astype(np.float64); vals=[]
    for row,state in zip(runs,states):
        m=r4m.BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda(); m.load_state_dict(state); pred,a=r4m.predict(m,Xt,Lt,Ut); met=r4e.metrics(pred,yt,short,reverse); row['test']=met; row['alpha_mean']=float(a.mean()); row['alpha_std']=float(a.std()); vals.append(met['mre_percent'])
    rec={'status':'completed','dataset':args.dataset,'classification':'frozen-R4M sparse-landmark evaluation; no retuning','workload_sha256':sha(args.workload),'index_sha256':sha(args.index),'node_count':n,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'feature_dim':int(Xtr.shape[1]),'fixed_protocol':{'landmark_seed':LANDMARK_SEED,'seeds':list(SEEDS),'time_budget_seconds_per_seed':TIME_LIMIT_S,'batch':BATCH,'lr':LR,'loss':'relative SmoothL1 beta=0.02'},'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rec,indent=2)+'\n'); print('R4N_SPARSE_COMPLETE',json.dumps({'dataset':args.dataset,'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd'],'per_seed':vals}),flush=True)
if __name__=='__main__': main()

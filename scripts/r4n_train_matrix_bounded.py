"""Run the frozen R4M architecture on a pre-frozen directed workload backed by an exact distance matrix.

No architecture/hyperparameter is selected here. The Jinan-frozen settings are reused verbatim:
random32 directed landmarks, 64 float32/node, 139-D features, 1024->512 bounded alpha network,
relative SmoothL1 beta=.02, AdamW lr=1e-3, batch 16384, 300 s/seed, seeds 42/99/1234.
Validation selects checkpoints; test is accessed only after all seed checkpoints are frozen.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4m_bounded_landmarknn as r4m
import scripts.r4e_directed_landmark_residual as r4e

SEEDS=(42,99,1234);BATCH=16384;LR=1e-3;VAL_EVERY=100;TIME_LIMIT_S=300.0;LANDMARK_SEED=20260914

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--workload',type=Path,required=True);ap.add_argument('--matrix',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    z=np.load(args.workload);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'].astype(np.float32);n=len(coords)
    D=np.load(args.matrix,mmap_mode='r');
    if D.shape!=(n,n):raise AssertionError((D.shape,n))
    if not np.isfinite(D).all():raise RuntimeError('nonfinite all-pairs matrix')
    train_nodes=np.unique(train[:,:2].astype(np.int64));rng=np.random.default_rng(LANDMARK_SEED);landmarks=np.sort(rng.choice(train_nodes,size=32,replace=False))
    node=np.concatenate((np.asarray(D[landmarks,:]).T,np.asarray(D[:,landmarks])),axis=1).astype(np.float32)
    scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25))
    Xtr,Ltr,Utr,mu,sd=r4m.build_features(node,coords,train[:,:2],scale);Xv,Lv,Uv,_,_=r4m.build_features(node,coords,val[:,:2],scale,mu,sd);ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64)
    tol=1e-6
    if np.any(Ltr>ytr+tol) or np.any(Utr<ytr-tol) or np.any(Lv>yv+tol) or np.any(Uv<yv-tol):raise AssertionError('certified bounds violated')
    G=Utr-Ltr;mask=G>1e-12;astar=np.clip((ytr[mask]-Ltr[mask])/G[mask],0,1);weights=G[mask]/ytr[mask];alpha0=r4m.weighted_median(astar,weights)
    Xt=torch.as_tensor(Xv,device='cuda');Lt=torch.as_tensor(Lv,dtype=torch.float32,device='cuda');Gt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda');yt=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
    runs=[];states=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);rr=np.random.default_rng(seed+20260914)
        m=r4m.BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);best=np.inf;best_step=0;best_state=None;started=time.perf_counter();step=0
        while time.perf_counter()-started<TIME_LIMIT_S:
            step+=1;ix=rr.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device='cuda');l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device='cuda');g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device='cuda');y=torch.as_tensor(ytr[ix],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);p=l+m(x)*g;re=(p-y)/y;loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02);loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval();
                with torch.no_grad():pv=Lt+m(Xt)*Gt;vm=float(torch.mean(torch.abs(pv-yt)/yt)*100)
                m.train()
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%1000==0:print('R4N_MATRIX_PROGRESS',args.dataset,seed,step,best,best_step,flush=True)
        if best_state is None:raise RuntimeError('no validation checkpoint')
        runs.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'train_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in m.parameters())});states.append(best_state)
    # Test is accessed only after all seed states have been frozen.
    test=np.load(args.workload)['test'].copy();Xt,Lt,Ut,_,_=r4m.build_features(node,coords,test[:,:2],scale,mu,sd);yt=test[:,2].astype(np.float64);rev=np.asarray(D[test[:,1].astype(np.int64),test[:,0].astype(np.int64)],dtype=np.float64)
    vals=[]
    for row,state in zip(runs,states):
        m=r4m.BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda();m.load_state_dict(state);pred,a=r4m.predict(m,Xt,Lt,Ut);met=r4e.metrics(pred,yt,short,rev);row['test']=met;row['alpha_mean']=float(a.mean());row['alpha_std']=float(a.std());vals.append(met['mre_percent'])
    rec={'status':'completed','dataset':args.dataset,'classification':'frozen-R4M multi-dataset evaluation; no Jinan-driven retuning','workload_sha256':sha(args.workload),'matrix_sha256':sha(args.matrix),'node_count':n,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'feature_dim':int(Xtr.shape[1]),'fixed_protocol':{'landmark_seed':LANDMARK_SEED,'seeds':list(SEEDS),'time_budget_seconds_per_seed':TIME_LIMIT_S,'batch':BATCH,'lr':LR,'loss':'relative SmoothL1 beta=0.02'},'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rec,indent=2)+'\n');print('R4N_MATRIX_COMPLETE',json.dumps({'dataset':args.dataset,'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd'],'per_seed':vals,'val':[r['best_validation_mre_percent'] for r in runs]}),flush=True)
if __name__=='__main__':main()

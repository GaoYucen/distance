"""R4G: bounded directed-distance residual with an MRE-equivalent alpha objective.

For G=U-L and alpha*=(d-L)/G, d=L+alpha*G and d_hat=L+alpha G imply
|d_hat-d|/d = (G/d)|alpha-alpha*|.  Therefore weighted alpha-L1 is exactly
sample MRE wherever G>0; G=0 contributes zero.  Sigmoid output preserves L<=d_hat<=U.

Development study only: the R4E test split has already been inspected.  All variants below are
predeclared before this run and selected only by validation MRE; test evaluation occurs after all
variant/seed checkpoints are frozen.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
from scripts.r4e_build_directed_jinan_workload import label_pairs

REPORT=ROOT/'reports/audit-r4g-20260914';RESULT=ROOT/'results/audit-r4g-20260914'
OUT=REPORT/'bounded_alpha_study.json';SEEDS=(42,99,1234);UPDATES=5000;BATCH=16384;VAL_EVERY=25;LR=1e-3


def rich_features(node,coords,ids,scale):
    base,L,U=r4e.pair_features(node,coords,ids,scale);ids=np.asarray(ids,dtype=np.int64);u,v=ids.T
    x=np.asarray(node,dtype=np.float32);xy=np.asarray(coords,dtype=np.float32)
    rawu=x[u].astype(np.float64)/scale; rawv=x[v].astype(np.float64)/scale
    # Absolute coordinate information is graph metadata and is available for every node at index build time.
    mu=xy.mean(0,dtype=np.float64);sd=xy.std(0,dtype=np.float64);sd=np.where(sd<1e-8,1.,sd)
    cu=(xy[u].astype(np.float64)-mu)/sd;cv=(xy[v].astype(np.float64)-mu)/sd;cd=cv-cu
    dot=np.sum(rawu*rawv,1);den=np.maximum(np.sqrt(np.sum(rawu*rawu,1)*np.sum(rawv*rawv,1)),1e-12);cos=(dot/den)[:,None]
    extra=np.concatenate((rawu,rawv,cu,cv,cd,np.abs(cd),cos),axis=1).astype(np.float32)
    return np.concatenate((base,extra),axis=1).astype(np.float32),L,U


def alpha_target(L,U,y):
    g=np.asarray(U)-np.asarray(L);y=np.asarray(y);mask=g>1e-12
    a=np.zeros_like(y,dtype=np.float64);a[mask]=(y[mask]-L[mask])/g[mask];a=np.clip(a,0.,1.)
    w=np.zeros_like(y,dtype=np.float64);w[mask]=g[mask]/y[mask]
    return a.astype(np.float32),w.astype(np.float32)


class Small(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)
class Large(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)


def predict_alpha(m,X):
    ans=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):ans.append(m(torch.as_tensor(X[i:i+32768],device='cuda')).cpu().numpy())
    return np.concatenate(ans)


def train_variant(name,cls,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv):
    at,wt=alpha_target(Ltr,Utr,ytr);rows=[];states=[]
    Xv_t=torch.as_tensor(Xv,device='cuda');Lv_t=torch.as_tensor(Lv,dtype=torch.float32,device='cuda');Gv_t=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda');yv_t=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed+20260914)
        m=cls(Xtr.shape[1]).cuda();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);best=np.inf;best_step=0;best_state=None;hist=[];started=time.perf_counter()
        for step in range(1,UPDATES+1):
            ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device='cuda');a=torch.as_tensor(at[ix],device='cuda');w=torch.as_tensor(wt[ix],device='cuda')
            opt.zero_grad(set_to_none=True);ph=m(x);loss=torch.mean(w*torch.abs(ph-a));loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad():pv=Lv_t+m(Xv_t)*Gv_t;vm=float(torch.mean(torch.abs(pv-yv_t)/yv_t)*100)
                m.train();hist.append({'step':step,'validation_mre_percent':vm,'train_exact_weighted_alpha_l1':float(loss.detach())})
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%500==0:print('R4G_PROGRESS',name,seed,step,best,best_step,flush=True)
        rows.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'history':hist,'train_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in m.parameters())});states.append(best_state)
    return rows,states


def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25));node,landmarks,A,_=r4e.build_index(z)
    Xo,Ltr,Utr=r4e.pair_features(node,coords,train[:,:2],scale);Xov,Lv,Uv=r4e.pair_features(node,coords,val[:,:2],scale)
    Xr,Lr,Ur=rich_features(node,coords,train[:,:2],scale);Xrv,Lrv,Urv=rich_features(node,coords,val[:,:2],scale)
    assert np.array_equal(Ltr,Lr) and np.array_equal(Utr,Ur) and np.array_equal(Lv,Lrv) and np.array_equal(Uv,Urv)
    ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64)
    specs=[('old109_exactMRE',Small,Xo,Xov),('rich246_small_exactMRE',Small,Xr,Xrv),('rich246_large_exactMRE',Large,Xr,Xrv)]
    trained={}
    for name,cls,X,Xv in specs:
        rows,states=train_variant(name,cls,X,Ltr,Utr,ytr,Xv,Lv,Uv,yv);trained[name]=(cls,X.shape[1],rows,states)
    # Freeze all variant and seed selections before loading/evaluating test labels.
    test=np.load(r4e.DATA)['test'].copy();yt=test[:,2].astype(np.float64);reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);Xot,Lt,Ut=r4e.pair_features(node,coords,test[:,:2],scale);Xrt,Lrt,Urt=rich_features(node,coords,test[:,:2],scale);assert np.array_equal(Lt,Lrt) and np.array_equal(Ut,Urt)
    ret={};outdir=RESULT/'bounded_alpha';outdir.mkdir(parents=True,exist_ok=True)
    for name,(cls,d,rows,states) in trained.items():
        Xt=Xot if name.startswith('old109') else Xrt;vals=[]
        for row,state in zip(rows,states):
            m=cls(d).cuda();m.load_state_dict(state);a=predict_alpha(m,Xt);p=Lt+a*(Ut-Lt);met=r4e.metrics(p,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent']);ck=outdir/f"{name}_seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'input_dim':d},ck);row['checkpoint']=str(ck.relative_to(ROOT))
        ret[name]={'feature_dim':d,'runs':rows,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    rec={'status':'completed','classification':'development-only; test previously exposed in R4E/R4F','identity':'MRE=(U-L)/d * |alpha-alpha_star|','index_scalars_per_node':64,'index_bytes_per_node_float32':256,'updates':UPDATES,'variants':ret}
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4G_COMPLETE',json.dumps({k:{'mean':v['test_mre_mean'],'sd':v['test_mre_sd'],'val':[r['best_validation_mre_percent'] for r in v['runs']],'steps':[r['best_step'] for r in v['runs']]} for k,v in ret.items()}),flush=True)
if __name__=='__main__':main()

"""R4M: LandmarkNN-capacity alpha predictor constrained to certified directed landmark bounds.

Uses the same random32 forward/reverse landmark index as R4E/R4F (64 float32 scalars/node) and the
survey-style normalized LandmarkNN pair features.  Five standardized bound-summary features are added.
The network has the same 1024-512 hidden widths as the R4F LandmarkNN adaptation, but its scalar output
passes through sigmoid and is decoded as L + alpha*(U-L), so every prediction remains in the certified
interval.  The final layer is initialized at the train-only weighted-median constant alpha, and training
uses the stable relative SmoothL1 objective from R4F.  Three seeds get the same 300-second per-seed wall
clock budget as the directed LandmarkNN baseline.  Validation selects checkpoints; the already-exposed
R4E test is read only after all choices are frozen.  Development study only.
"""
from __future__ import annotations
import copy,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
import scripts.r4f_directed_strong_baselines as r4f
from scripts.r4e_build_directed_jinan_workload import label_pairs

REPORT=ROOT/'reports/audit-r4m-20260915';RESULT=ROOT/'results/audit-r4m-20260915';OUT=REPORT/'bounded_landmarknn.json'
SEEDS=(42,99,1234);BATCH=16384;LR=1e-3;VAL_EVERY=100;TIME_LIMIT_S=300.0

def weighted_median(v,w):
    o=np.argsort(v,kind='stable');v=v[o];w=w[o];c=np.cumsum(w);return float(v[np.searchsorted(c,.5*c[-1],side='left')])

def bound_extra(L,U,scale):
    L=np.asarray(L,dtype=np.float64);U=np.asarray(U,dtype=np.float64);G=np.maximum(U-L,0.);eps=1e-9
    return np.column_stack((np.log1p(L/scale),np.log1p(U/scale),np.log1p(G/scale),L/np.maximum(U,eps),G/np.maximum(U,eps))).astype(np.float32)

def standardize_fit(x):
    mu=x.mean(0,dtype=np.float64);sd=x.std(0,dtype=np.float64);sd=np.where(sd<1e-8,1.,sd);return mu,sd

def standardize_apply(x,mu,sd):return ((x.astype(np.float64)-mu)/sd).astype(np.float32)

def build_features(node,coords,pairs,scale,mu=None,sd=None):
    base,_,l2=r4f.encode_numpy(node,coords,pairs,True);_,L,U=r4e.pair_features(node,coords,pairs,scale);e=bound_extra(L,U,scale)
    if mu is None:mu,sd=standardize_fit(e)
    e=standardize_apply(e,mu,sd);return np.concatenate((base,l2,e),axis=1).astype(np.float32),L,U,mu,sd

class BoundedLandmarkNN(nn.Module):
    def __init__(self,d,alpha0):
        super().__init__();self.fc1=nn.Linear(d,1024);self.fc2=nn.Linear(1024,512);self.fc3=nn.Linear(512,1)
        a=float(np.clip(alpha0,1e-5,1-1e-5));nn.init.zeros_(self.fc3.weight);nn.init.constant_(self.fc3.bias,math.log(a/(1-a)))
    def forward(self,x):
        x=F.relu(self.fc1(x));x=F.relu(self.fc2(x));return torch.sigmoid(self.fc3(x)).squeeze(-1)

def predict(m,X,L,U):
    aa=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):aa.append(m(torch.as_tensor(X[i:i+32768],device='cuda')).cpu().numpy())
    a=np.concatenate(aa);return L+a*(U-L),a

def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25));node,landmarks,A,build_seconds=r4e.build_index(z)
    Xtr,Ltr,Utr,mu,sd=build_features(node,coords,train[:,:2],scale);Xv,Lv,Uv,_,_=build_features(node,coords,val[:,:2],scale,mu,sd);ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64)
    G=Utr-Ltr;mask=G>1e-12;astar=np.clip((ytr[mask]-Ltr[mask])/G[mask],0,1);weights=G[mask]/ytr[mask];alpha0=weighted_median(astar,weights)
    init_val=Lv+alpha0*(Uv-Lv);init_val_mre=float(100*np.mean(np.abs(init_val-yv)/yv));print('R4M_INIT',alpha0,init_val_mre,Xtr.shape[1],flush=True)
    runs=[];states=[]
    Xt=torch.as_tensor(Xv,device='cuda');Lt=torch.as_tensor(Lv,dtype=torch.float32,device='cuda');Gt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda');yt=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed+20260914);m=BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);best=np.inf;best_step=0;best_state=None;hist=[];started=time.perf_counter();step=0
        while time.perf_counter()-started<TIME_LIMIT_S:
            step+=1;ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device='cuda');l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device='cuda');g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device='cuda');y=torch.as_tensor(ytr[ix],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);p=l+m(x)*g;re=(p-y)/y;loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02);loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad():pv=Lt+m(Xt)*Gt;vm=float(torch.mean(torch.abs(pv-yt)/yt)*100)
                m.train();elapsed=time.perf_counter()-started;hist.append({'step':step,'elapsed_seconds':elapsed,'validation_mre_percent':vm,'train_relative_smoothl1':float(loss.detach())})
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
                if step%1000==0:print('R4M_PROGRESS',seed,step,elapsed,best,best_step,flush=True)
        if best_state is None:raise RuntimeError('no validation checkpoint')
        runs.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'history':hist,'train_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in m.parameters())});states.append(best_state)
    # Freeze all seed checkpoints before development test access.
    test=np.load(r4e.DATA)['test'].copy();Xt,Lt,Ut,_,_=build_features(node,coords,test[:,:2],scale,mu,sd);yt=test[:,2].astype(np.float64);reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);vals=[];outdir=RESULT/'bounded_landmarknn';outdir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=BoundedLandmarkNN(Xtr.shape[1],alpha0).cuda();m.load_state_dict(state);pred,a=predict(m,Xt,Lt,Ut);met=r4e.metrics(pred,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent']);ck=outdir/f"seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'input_dim':Xtr.shape[1],'bound_extra_mu':mu,'bound_extra_sd':sd,'alpha0':alpha0},ck);row['checkpoint']=str(ck.relative_to(ROOT))
    rec={'status':'completed','classification':'development-only native bounded LandmarkNN-capacity study; current test already exposed','index_scalars_per_node':64,'index_bytes_per_node_float32':256,'landmarks':landmarks.tolist(),'index_build_seconds':build_seconds,'feature_dim':int(Xtr.shape[1]),'alpha0_train_weighted_median':alpha0,'initial_validation_mre_percent':init_val_mre,'time_budget_seconds_per_seed':TIME_LIMIT_S,'loss':'relative SmoothL1 beta=0.02','runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rec,indent=2)+'\n');print('R4M_COMPLETE',json.dumps({'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd'],'val':[x['best_validation_mre_percent'] for x in runs],'steps':[x['best_step'] for x in runs]}),flush=True)
if __name__=='__main__':main()

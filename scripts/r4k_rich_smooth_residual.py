"""R4K: bounded residual with richer pair features and the stable R4F relative-SmoothL1 objective.

R4G showed that replacing the R4F objective with exact weighted alpha-L1 destabilized training, so this
follow-up changes only the information available to the alpha predictor while restoring the R4F loss.
The random32 directed landmark index is retained because the validation-selected final residual in R4F
was better than the coord-FPS32 residual in R4I, even though coord-FPS tightened the lower bound.

Two predeclared capacities share exactly the same 246-dimensional features and 64-float/node index.
All checkpoint choices use validation MRE; the already-exposed R4E test is read only after every
variant/seed checkpoint is frozen.  This is development evidence, not final confirmation.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
import scripts.r4g_bounded_alpha as r4g
from scripts.r4e_build_directed_jinan_workload import label_pairs

REPORT=ROOT/'reports/audit-r4k-20260915';RESULT=ROOT/'results/audit-r4k-20260915';OUT=REPORT/'rich_smooth_residual.json'
SEEDS=(42,99,1234);UPDATES=5000;BATCH=16384;VAL_EVERY=25;LR=1e-3

class Small(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)
class Medium(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)

def predict(m,X,L,U):
    aa=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):aa.append(m(torch.as_tensor(X[i:i+32768],device='cuda')).cpu().numpy())
    a=np.concatenate(aa);return L+a*(U-L),a

def train_variant(name,cls,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv):
    rows=[];states=[];Xt=torch.as_tensor(Xv,device='cuda');Lt=torch.as_tensor(Lv,dtype=torch.float32,device='cuda');Gt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda');yt=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed+20260914)
        m=cls(Xtr.shape[1]).cuda();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);best=np.inf;best_step=0;best_state=None;hist=[];started=time.perf_counter()
        for step in range(1,UPDATES+1):
            ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device='cuda');l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device='cuda');g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device='cuda');y=torch.as_tensor(ytr[ix],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);p=l+m(x)*g;re=(p-y)/y;loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02);loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad():pv=Lt+m(Xt)*Gt;vm=float(torch.mean(torch.abs(pv-yt)/yt)*100)
                m.train();hist.append({'step':step,'validation_mre_percent':vm,'train_relative_smoothl1':float(loss.detach())})
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%500==0:print('R4K_PROGRESS',name,seed,step,best,best_step,flush=True)
        if best_state is None:raise RuntimeError('no checkpoint')
        rows.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'history':hist,'train_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in m.parameters())});states.append(best_state)
    return rows,states

def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25));node,landmarks,A,_=r4e.build_index(z)
    Xtr,Ltr,Utr=r4g.rich_features(node,coords,train[:,:2],scale);Xv,Lv,Uv=r4g.rich_features(node,coords,val[:,:2],scale);ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64)
    if Xtr.shape[1]!=246:raise AssertionError(Xtr.shape)
    specs=[('rich246_small_smooth',Small),('rich246_medium_smooth',Medium)];trained={}
    for name,cls in specs:
        rows,states=train_variant(name,cls,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv);trained[name]=(cls,rows,states)
    # Freeze both variants/all seed checkpoints before accessing the development test.
    test=np.load(r4e.DATA)['test'].copy();Xt,Lt,Ut=r4g.rich_features(node,coords,test[:,:2],scale);yt=test[:,2].astype(np.float64);reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);ret={};outdir=RESULT/'rich_smooth';outdir.mkdir(parents=True,exist_ok=True)
    for name,(cls,rows,states) in trained.items():
        vals=[]
        for row,state in zip(rows,states):
            m=cls(Xtr.shape[1]).cuda();m.load_state_dict(state);pred,a=predict(m,Xt,Lt,Ut);met=r4e.metrics(pred,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent']);ck=outdir/f"{name}_seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'input_dim':Xtr.shape[1]},ck);row['checkpoint']=str(ck.relative_to(ROOT))
        ret[name]={'feature_dim':int(Xtr.shape[1]),'runs':rows,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    rec={'status':'completed','classification':'development-only rich-feature bounded residual; test previously exposed','index_scalars_per_node':64,'index_bytes_per_node_float32':256,'loss':'relative SmoothL1 beta=0.02','updates':UPDATES,'variants':ret}
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4K_COMPLETE',json.dumps({k:{'mean':v['test_mre_mean'],'sd':v['test_mre_sd'],'val':[x['best_validation_mre_percent'] for x in v['runs']],'steps':[x['best_step'] for x in v['runs']]} for k,v in ret.items()}),flush=True)
if __name__=='__main__':main()

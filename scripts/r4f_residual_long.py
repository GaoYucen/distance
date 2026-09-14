"""R4F: validation-only longer training of the same R4E MLPAlpha residual architecture.

No architecture/feature/landmark change. The only intervention is a pre-declared increase from 1500 to
5000 updates. Current R4E test is already a development benchmark; this run must not be described as
fresh confirmation. Best checkpoint is selected by validation MRE, then test is evaluated once per seed.
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
from scripts.r4e_build_directed_jinan_workload import label_pairs

REPORT=ROOT/'reports/audit-r4f-20260914';RESULT=ROOT/'results/audit-r4f-20260914';OUT=REPORT/'directed_residual_mlp_5000.json'
SEEDS=(42,99,1234);UPDATES=5000;BATCH=16384;VAL_EVERY=25;LR=1e-3


def predict(m,X,L,U):
    a=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):a.append(m(torch.as_tensor(X[i:i+32768],device='cuda')).cpu().numpy())
    a=np.concatenate(a);return L+a*(U-L),a


def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25))
    node,landmarks,A,build_seconds=r4e.build_index(z)
    Xtr,Ltr,Utr=r4e.pair_features(node,coords,train[:,:2],scale);Xv,Lv,Uv=r4e.pair_features(node,coords,val[:,:2],scale)
    ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64);runs=[];states=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed+20260914)
        m=r4e.AlphaMLP(Xtr.shape[1]).cuda();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5)
        Xt=torch.as_tensor(Xv,device='cuda');Lt=torch.as_tensor(Lv,dtype=torch.float32,device='cuda');Gt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device='cuda');yt=torch.as_tensor(yv,dtype=torch.float32,device='cuda')
        best=np.inf;best_step=0;best_state=None;hist=[];started=time.perf_counter()
        for step in range(1,UPDATES+1):
            ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device='cuda');l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device='cuda');g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device='cuda');y=torch.as_tensor(ytr[ix],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);p=l+m(x)*g;re=(p-y)/y;loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02);loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad():pv=Lt+m(Xt)*Gt;vm=float(torch.mean(torch.abs(pv-yt)/yt)*100)
                m.train();hist.append({'step':step,'validation_mre_percent':vm,'train_relative_smoothl1':float(loss.detach())})
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%500==0:print('R4F_RESIDUAL_LONG_PROGRESS',seed,step,best,best_step,flush=True)
        runs.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'history':hist,'train_seconds':time.perf_counter()-started});states.append(best_state)
    # All choices frozen; test is a development evaluation only because R4E already exposed it.
    test=np.load(r4e.DATA)['test'].copy();Xt,Lt,Ut=r4e.pair_features(node,coords,test[:,:2],scale);yt=test[:,2].astype(np.float64);reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);vals=[]
    outdir=RESULT/'directed_residual_mlp_5000';outdir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=r4e.AlphaMLP(Xtr.shape[1]).cuda();m.load_state_dict(state);pred,a=predict(m,Xt,Lt,Ut);met=r4e.metrics(pred,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent'])
        p=outdir/f"seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'scale':scale,'input_dim':Xtr.shape[1]},p);row['checkpoint']=str(p.relative_to(ROOT))
    rec={'status':'completed','classification':'development-only longer-training diagnostic; not fresh confirmation','change_from_r4e':'updates 1500 -> 5000 only',
         'updates':UPDATES,'seeds':list(SEEDS),'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'feature_dim':int(Xtr.shape[1]),
         'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4F_RESIDUAL_LONG_COMPLETE',json.dumps({'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd'],'best_val':[r['best_validation_mre_percent'] for r in runs],'steps':[r['best_step'] for r in runs]}),flush=True)
if __name__=='__main__':main()

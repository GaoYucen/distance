"""R4F: survey-style strong baselines on the survey-matched native-directed Jinan workload.

This is a directed adaptation, not a claim that the original survey methods natively support directed graphs.
It reuses the same 32 directed landmarks (forward+reverse = 64 scalars/node) as R4E and mirrors
survey CatBoost/LandmarkNN feature construction: src/dst landmark features, src/dst coordinates,
landmark cosine similarity, and coordinate distance.

Train/validation labels may be used for fitting/model selection. Test labels are accessed only after
selection/training is frozen inside each model run. Current R4E test is a development benchmark, not a
fresh final confirmation set.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
from scripts.r4e_build_directed_jinan_workload import label_pairs

DATA=r4e.DATA; REPORT=ROOT/'reports/audit-r4f-20260914'; RESULT=ROOT/'results/audit-r4f-20260914'
SEEDS=(42,99,1234); BATCH=16384; LR=1e-3; TIME_LIMIT_S=300.0


def metric(pred,y,short,reverse):
    return r4e.metrics(np.asarray(pred,dtype=np.float64),np.asarray(y,dtype=np.float64),short,reverse)


def node_stats(a):
    mu=a.mean(0,keepdims=True); sd=a.std(0,keepdims=True); sd=np.where(sd<1e-8,1.,sd); return mu,sd


def encode_numpy(node,coords,pairs,normalized=False):
    pairs=np.asarray(pairs,dtype=np.int64);u,v=pairs.T
    land=np.asarray(node,dtype=np.float32); xy=np.asarray(coords,dtype=np.float32)
    if normalized:
        lm,ls=node_stats(land);cm,cs=node_stats(xy);land=(land-lm)/ls;xy=(xy-cm)/cs
    a,b=land[u],land[v];ca,cb=xy[u],xy[v]
    dot=np.sum(a*b,axis=1);den=np.maximum(np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1)),1e-12)
    cos=(dot/den)[:,None].astype(np.float32)
    # Survey CatBoost code uses coordinate L1; LandmarkNN code uses coordinate L2.
    l1=np.abs(ca-cb).sum(1,keepdims=True).astype(np.float32)
    l2=np.sqrt(np.square(ca-cb).sum(1,keepdims=True)).astype(np.float32)
    base=np.concatenate((a,b,ca,cb,cos),axis=1).astype(np.float32)
    return base,l1,l2


def build_common():
    z=np.load(DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];
    node,landmarks,A,build_seconds=r4e.build_index(z)
    short=float(np.quantile(train[:,2],.25))
    return z,train,val,coords,node,landmarks,A,short,build_seconds


def run_catboost():
    from catboost import CatBoostRegressor, Pool
    out=REPORT/'directed_catboost_alt32.json'
    if out.exists(): raise FileExistsError(out)
    z,train,val,coords,node,landmarks,A,short,build_seconds=build_common()
    b,l1,_=encode_numpy(node,coords,train[:,:2],False);Xtr=np.concatenate((b,l1),1);ytr=train[:,2].astype(np.float64)
    b,l1,_=encode_numpy(node,coords,val[:,:2],False);Xv=np.concatenate((b,l1),1);yv=val[:,2].astype(np.float64)
    val_pool=Pool(Xv,yv); train_pool=Pool(Xtr,ytr)
    # Mirror the survey implementation: RMSE objective, MAPE monitor, CPU, 500-tree chunks, 5-minute budget.
    current=None; total=0; started=time.perf_counter(); history=[]
    while time.perf_counter()-started < TIME_LIMIT_S:
        chunk=500
        m=CatBoostRegressor(iterations=chunk,learning_rate=.1,random_seed=1234,loss_function='RMSE',eval_metric='MAPE',
                            task_type='CPU',thread_count=4,verbose=False)
        m.fit(train_pool,init_model=current,eval_set=val_pool,verbose=False)
        current=m;total+=chunk
        pv=current.predict(Xv); vm=float(100*np.mean(np.abs(pv-yv)/yv));elapsed=time.perf_counter()-started
        history.append({'trees':total,'elapsed_seconds':elapsed,'validation_mre_percent':vm})
        print('R4F_CATBOOST_PROGRESS',total,elapsed,vm,flush=True)
        if elapsed>=TIME_LIMIT_S: break
    # Fixed time-budget model; only now access test labels.
    test=np.load(DATA)['test'].copy();b,l1,_=encode_numpy(node,coords,test[:,:2],False);Xt=np.concatenate((b,l1),1);yt=test[:,2].astype(np.float64)
    reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);pred=current.predict(Xt)
    met=metric(pred,yt,short,reverse)
    RESULT.mkdir(parents=True,exist_ok=True);model_path=RESULT/'directed_catboost_alt32.cbm';current.save_model(model_path)
    rec={'status':'completed','classification':'directed adaptation of survey CatBoost feature recipe; development benchmark',
         'survey_reference':'purduedb/shortest-distance-survey@dcaa89d38300bfb823eda84ccdfc85c42edbeae8 src/models/catboostmodel.py',
         'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'landmarks':landmarks.tolist(),'index_build_seconds':build_seconds,
         'feature_dim':int(Xtr.shape[1]),'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
         'time_budget_seconds':TIME_LIMIT_S,'trees':total,'history':history,'test':met,'model_path':str(model_path.relative_to(ROOT))}
    REPORT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4F_CATBOOST_COMPLETE',json.dumps({'mre':met['mre_percent'],'short':met['short_mre_percent'],'asym':met['high_asymmetry_mre_percent'],'trees':total}),flush=True)


class LandmarkNN(nn.Module):
    def __init__(self,d,max_distance):
        super().__init__(); self.max_distance=float(max_distance)
        self.net=nn.Sequential(nn.Linear(d,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,1))
    def forward(self,x): return self.net(x).squeeze(-1)*self.max_distance


def predict_nn(m,X,device):
    out=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):out.append(m(torch.as_tensor(X[i:i+32768],device=device)).cpu().numpy())
    return np.concatenate(out)


def run_landmarknn():
    out=REPORT/'directed_landmarknn_alt32.json'
    if out.exists(): raise FileExistsError(out)
    z,train,val,coords,node,landmarks,A,short,build_seconds=build_common()
    b,_,l2=encode_numpy(node,coords,train[:,:2],True);Xtr=np.concatenate((b,l2),1);ytr=train[:,2].astype(np.float32)
    b,_,l2=encode_numpy(node,coords,val[:,:2],True);Xv=np.concatenate((b,l2),1);yv=val[:,2].astype(np.float32)
    maxd=float(np.max(ytr));device='cuda'; runs=[];states=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
        m=LandmarkNN(Xtr.shape[1],maxd).to(device);opt=torch.optim.Adam(m.parameters(),lr=LR);rng=np.random.default_rng(seed+20260914)
        best=np.inf;best_state=None;best_step=0;hist=[];started=time.perf_counter();step=0
        while time.perf_counter()-started < TIME_LIMIT_S:
            step+=1;ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device=device);y=torch.as_tensor(ytr[ix],device=device)
            opt.zero_grad(set_to_none=True);pred=m(x);loss=F.mse_loss(pred/maxd,y/maxd);loss.backward();opt.step()
            if step%100==0:
                pv=predict_nn(m,Xv,device);vm=float(100*np.mean(np.abs(pv-yv)/yv));elapsed=time.perf_counter()-started
                hist.append({'step':step,'elapsed_seconds':elapsed,'validation_mre_percent':vm})
                if vm<best:best=vm;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
                print('R4F_LANDMARKNN_PROGRESS',seed,step,elapsed,best,flush=True)
        if best_state is None: raise RuntimeError('no validation checkpoint')
        runs.append({'seed':seed,'best_step':best_step,'best_validation_mre_percent':best,'history':hist,'train_seconds':time.perf_counter()-started});states.append(best_state)
    # All choices frozen; only now access test labels.
    test=np.load(DATA)['test'].copy();b,_,l2=encode_numpy(node,coords,test[:,:2],True);Xt=np.concatenate((b,l2),1);yt=test[:,2].astype(np.float64)
    reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);vals=[];outdir=RESULT/'directed_landmarknn_alt32';outdir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=LandmarkNN(Xtr.shape[1],maxd).to(device);m.load_state_dict(state);pred=predict_nn(m,Xt,device);met=metric(pred,yt,short,reverse);row['test']=met;vals.append(met['mre_percent'])
        p=outdir/f"seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'input_dim':Xtr.shape[1],'max_distance':maxd},p);row['checkpoint']=str(p.relative_to(ROOT))
    rec={'status':'completed','classification':'directed adaptation of survey LandmarkNN architecture; development benchmark',
         'survey_reference':'purduedb/shortest-distance-survey@dcaa89d38300bfb823eda84ccdfc85c42edbeae8 src/models/catboostnn.py',
         'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'landmarks':landmarks.tolist(),'index_build_seconds':build_seconds,
         'feature_dim':int(Xtr.shape[1]),'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'time_budget_seconds_per_seed':TIME_LIMIT_S,
         'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    REPORT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4F_LANDMARKNN_COMPLETE',json.dumps({'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd']}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['catboost','landmarknn'],required=True);a=ap.parse_args()
    run_catboost() if a.model=='catboost' else run_landmarknn()
if __name__=='__main__':main()

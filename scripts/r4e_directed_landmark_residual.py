"""R4E: bound-preserving residual decoders over a 32-landmark directed index.

Development protocol only. Train labels fit parameters; validation MRE selects
checkpoints/global alpha; test labels are read only after selections are frozen.
"""
from __future__ import annotations
import copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4c_realroads as r4c
from scripts.r4e_build_directed_jinan_workload import label_pairs

DATA=ROOT/'data/protocol_r4e/Jinan_native_directed_workload_500k.npz'
RAW=ROOT/'data/figshare_native_20260913/edge_jinan.csv'
REPORT=ROOT/'reports/audit-r4e-20260914'
RESULT=ROOT/'results/audit-r4e-20260914'
SEEDS=(42,99,1234);UPDATES=1500;BATCH=16384;VAL_EVERY=25;LR=1e-3


def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(pred,y,short_threshold,reverse=None):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64);rel=np.abs(p-y)/y
    out={'count':len(y),'mre_percent':float(100*rel.mean()),'mae_m':float(np.abs(p-y).mean()),
         'p95_relative_percent':float(100*np.quantile(rel,.95)),'max_relative_percent':float(100*rel.max()),
         'short_count':int(np.sum(y<=short_threshold)),'short_mre_percent':float(100*rel[y<=short_threshold].mean()) if np.any(y<=short_threshold) else None,
         'overprediction_fraction':float(np.mean(p>y+1e-6)),'underprediction_fraction':float(np.mean(p<y-1e-6))}
    if reverse is not None:
        reverse=np.asarray(reverse,dtype=np.float64);alpha=np.abs(y-reverse)/((y+reverse)/2);mask=alpha>=.2
        out['high_asymmetry_count']=int(mask.sum());out['high_asymmetry_mre_percent']=float(100*rel[mask].mean()) if mask.any() else None
    return out


def weighted_median(values,weights):
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order];c=np.cumsum(w)
    return float(v[np.searchsorted(c,.5*c[-1],side='left')])


def build_index(z):
    train=z['train'];n=len(z['coordinates']);train_nodes=np.unique(train[:,:2].astype(np.int64));rng=np.random.default_rng(20260914)
    landmarks=np.sort(rng.choice(train_nodes,size=32,replace=False));A=r4c.build_native_csr('Jinan_native_directed',z,RAW)
    started=time.perf_counter();fr=dijkstra(A,directed=True,indices=landmarks);to=dijkstra(A.T,directed=True,indices=landmarks)
    if not np.isfinite(fr).all() or not np.isfinite(to).all():raise RuntimeError('nonfinite landmark distances')
    node=np.concatenate((fr.T,to.T),axis=1).astype(np.float32)
    return node,landmarks,A,time.perf_counter()-started


def pair_features(node,coords,ids,scale):
    ids=np.asarray(ids,dtype=np.int64);u,v=ids.T
    # Central float32 values are used as model features. For certified interval endpoints,
    # outward-round each stored coordinate by one float32 representable value so the
    # original float64 Dijkstra distance is bracketed after round-to-nearest storage.
    x=np.asarray(node,dtype=np.float32)
    f=x[:,:32].astype(np.float64);t=x[:,32:].astype(np.float64)
    low1=f[v]-f[u];low2=t[u]-t[v];low=np.concatenate((low1,low2),axis=1)
    up=t[u]+f[v]
    lo=np.nextafter(x,np.float32(-np.inf),dtype=np.float32).astype(np.float64)
    hi=np.nextafter(x,np.float32(np.inf),dtype=np.float32).astype(np.float64)
    flo,fhi=lo[:,:32],hi[:,:32];tlo,thi=lo[:,32:],hi[:,32:]
    low_safe=np.concatenate((flo[v]-fhi[u],tlo[u]-thi[v]),axis=1)
    up_safe=thi[u]+fhi[v]
    L=np.maximum(0.,low_safe.max(1));U=up_safe.min(1)
    if np.any(U<L):raise AssertionError(('invalid landmark bounds after outward rounding',float(np.max(L-U))))
    dxy=coords[v].astype(np.float64)-coords[u].astype(np.float64);eu=np.sqrt(np.square(dxy).sum(1));adx=np.abs(dxy)
    top=np.partition(low,-4,axis=1)[:,-4:];bottom=np.partition(up,3,axis=1)[:,:4]
    eps=1e-6
    summary=np.column_stack((L,U,U-L,eu,adx[:,0],adx[:,1],top.max(1),np.mean(top,1),bottom.min(1),np.mean(bottom,1),
                             L/np.maximum(U,eps),eu/np.maximum(U,eps),(U-L)/np.maximum(U,eps)))
    X=np.concatenate((low/scale,up/scale,summary[:,:10]/scale,summary[:,10:]),axis=1).astype(np.float32)
    return X,L.astype(np.float64),U.astype(np.float64)


class AlphaLinear(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Linear(d,1)
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)

class AlphaMLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(-1)


def train_family(name,cls,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv,device):
    rows=[];states=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
        m=cls(Xtr.shape[1]).to(device);opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);rng=np.random.default_rng(seed+20260914)
        best=float('inf');best_step=-1;best_state=None;history=[];started=time.perf_counter()
        Xt=torch.as_tensor(Xv,device=device);Lt=torch.as_tensor(Lv,dtype=torch.float32,device=device);Gt=torch.as_tensor(Uv-Lv,dtype=torch.float32,device=device);yt=torch.as_tensor(yv,dtype=torch.float32,device=device)
        for step in range(1,UPDATES+1):
            ix=rng.integers(0,len(Xtr),size=BATCH);x=torch.as_tensor(Xtr[ix],device=device)
            l=torch.as_tensor(Ltr[ix],dtype=torch.float32,device=device);g=torch.as_tensor((Utr-Ltr)[ix],dtype=torch.float32,device=device);y=torch.as_tensor(ytr[ix],dtype=torch.float32,device=device)
            opt.zero_grad(set_to_none=True);pred=l+m(x)*g;re=(pred-y)/y
            loss=F.smooth_l1_loss(re,torch.zeros_like(re),beta=.02);loss.backward();opt.step()
            if step%VAL_EVERY==0:
                m.eval()
                with torch.no_grad():pv=Lt+m(Xt)*Gt;val=float(torch.mean(torch.abs(pv-yt)/yt)*100)
                m.train();history.append({'step':step,'validation_mre_percent':val,'train_relative_smoothl1':float(loss.detach())})
                if val<best:
                    best=val;best_step=step;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
            if step%250==0:print('R4E_RESIDUAL_PROGRESS',name,seed,step,best,flush=True)
        if best_state is None:raise RuntimeError('no checkpoint')
        rows.append({'seed':seed,'best_step':best_step,'best_validation_mre_percent':best,'history':history,'train_seconds':time.perf_counter()-started,
                     'parameter_count':sum(p.numel() for p in m.parameters())});states.append(best_state)
    return rows,states


def predict_state(cls,state,X,L,U,device):
    m=cls(X.shape[1]).to(device);m.load_state_dict(state);m.eval();parts=[]
    with torch.no_grad():
        for b in range(0,len(X),32768):
            x=torch.as_tensor(X[b:b+32768],device=device);a=m(x).cpu().numpy();parts.append(a)
    a=np.concatenate(parts);return L+a*(U-L),a


def main():
    if (REPORT/'directed_landmark_residual.json').exists():raise FileExistsError('R4E residual report exists')
    z=np.load(DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25))
    node,landmarks,A,build_seconds=build_index(z)
    Xtr,Ltr,Utr=pair_features(node,coords,train[:,:2],scale);Xv,Lv,Uv=pair_features(node,coords,val[:,:2],scale)
    ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64)
    for L,U,y,split in [(Ltr,Utr,ytr,'train'),(Lv,Uv,yv,'validation')]:
        tol=1e-9
        if np.any(L>y+tol) or np.any(U<y-tol):raise AssertionError(('bound violation',split,float(np.max(L-y)),float(np.max(y-U))))
    gap=Uv-Lv;mask=gap>1e-9;targets=np.clip((yv[mask]-Lv[mask])/gap[mask],0,1);weights=gap[mask]/yv[mask]
    alpha=weighted_median(targets,weights)
    val_global=Lv+alpha*(Uv-Lv)
    linear_rows,linear_states=train_family('LinearAlpha',AlphaLinear,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv,'cuda')
    mlp_rows,mlp_states=train_family('MLPAlpha',AlphaMLP,Xtr,Ltr,Utr,ytr,Xv,Lv,Uv,yv,'cuda')
    # Selection is frozen. Only now load test labels.
    zt=np.load(DATA);test=zt['test'].copy();Xt,Lt,Ut=pair_features(node,coords,test[:,:2],scale);yt=test[:,2].astype(np.float64)
    tol=1e-9
    if np.any(Lt>yt+tol) or np.any(Ut<yt-tol):raise AssertionError(('test bound violation',float(np.max(Lt-yt)),float(np.max(yt-Ut))))
    reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1])
    baselines={'ALT32-LB':metrics(Lt,yt,short,reverse),'ALT32-UB':metrics(Ut,yt,short,reverse),
               'GlobalAlpha':{'alpha':alpha,**metrics(Lt+alpha*(Ut-Lt),yt,short,reverse),
                              'validation_mre_percent':float(100*np.mean(np.abs(val_global-yv)/yv))}}
    learned={}
    for name,cls,rows,states in [('LinearAlpha',AlphaLinear,linear_rows,linear_states),('MLPAlpha',AlphaMLP,mlp_rows,mlp_states)]:
        vals=[]
        outdir=RESULT/'directed_landmark_residual'/name;outdir.mkdir(parents=True,exist_ok=True)
        for row,state in zip(rows,states):
            pred,a=predict_state(cls,state,Xt,Lt,Ut,'cuda');met=metrics(pred,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent'])
            ck=outdir/f"seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'scale':scale,'input_dim':Xtr.shape[1]},ck);row['checkpoint']=str(ck.relative_to(ROOT));row['checkpoint_sha256']=sha(ck)
        learned[name]={'runs':rows,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    RESULT.mkdir(parents=True,exist_ok=True);idx=RESULT/'directed_landmark_residual/ALT32_index.npz';idx.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(idx,node_features=node,landmarks=landmarks)
    report={'status':'completed','classification':'survey-matched directed workload development; not final independent confirmation',
            'data_sha256':sha(DATA),'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,
            'index_build_seconds':build_seconds,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'scale_m':scale,'short_threshold_m':short,
            'feature_dim':int(Xtr.shape[1]),'bound_storage_policy':'float32 index with one-ULP outward rounding at decode for certified L<=d<=U',
            'fixed_protocol':{'updates':UPDATES,'batch':BATCH,'val_every':VAL_EVERY,'lr':LR,'seeds':list(SEEDS),'loss':'relative SmoothL1 beta=0.02'},
            'baselines':baselines,'learned':learned,'index_artifact':str(idx.relative_to(ROOT)),'index_sha256':sha(idx)}
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'directed_landmark_residual.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('R4E_RESIDUAL_COMPLETE',json.dumps({'baselines':baselines,'learned':{k:{'test_mre_mean':v['test_mre_mean'],'test_mre_sd':v['test_mre_sd']} for k,v in learned.items()}}),flush=True)

if __name__=='__main__':main()

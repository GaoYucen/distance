"""R4I: coord-FPS32 directed-landmark residual at the same 64-scalar/node budget.

This is a development follow-up selected from the validation-only R4H diagnostic, where coordinate
farthest-point landmarks improved the ALT lower-bound validation MRE over the frozen random32 baseline.
Architecture, loss, batch size, optimizer, and 5000-update budget match R4F residual-long. All three
seed checkpoints are selected only by validation MRE; the already-exposed R4E test is read only after
all selections are frozen, so this remains development evidence rather than final confirmation.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
from scripts.r4e_build_directed_jinan_workload import label_pairs
from scripts.r4h_landmark_selection_val import fps_order

REPORT=ROOT/'reports/audit-r4i-20260915';RESULT=ROOT/'results/audit-r4i-20260915';OUT=REPORT/'coordfps32_residual_mlp_5000.json'
SEEDS=(42,99,1234);UPDATES=5000;BATCH=16384;VAL_EVERY=25;LR=1e-3;K=32


def build_coordfps_index(z):
    train=z['train'];coords=z['coordinates'];train_nodes=np.unique(train[:,:2].astype(np.int64));landmarks=np.asarray(fps_order(coords,train_nodes),dtype=np.int64)[:K]
    A=r4e.r4c.build_native_csr('Jinan_native_directed',z,r4e.RAW);started=time.perf_counter();fr=dijkstra(A,directed=True,indices=landmarks);to=dijkstra(A.T,directed=True,indices=landmarks)
    if not np.isfinite(fr).all() or not np.isfinite(to).all():raise RuntimeError('nonfinite landmark distances')
    node=np.concatenate((fr.T,to.T),axis=1).astype(np.float32)
    return node,landmarks,A,time.perf_counter()-started


def predict(m,X,L,U):
    aa=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(X),32768):aa.append(m(torch.as_tensor(X[i:i+32768],device='cuda')).cpu().numpy())
    a=np.concatenate(aa);return L+a*(U-L),a


def assert_bounds(L,U,y,label):
    if np.any(L>y+1e-9) or np.any(U<y-1e-9):
        raise AssertionError((label,float(np.max(L-y)),float(np.max(y-U))))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25))
    node,landmarks,A,build_seconds=build_coordfps_index(z)
    Xtr,Ltr,Utr=r4e.pair_features(node,coords,train[:,:2],scale);Xv,Lv,Uv=r4e.pair_features(node,coords,val[:,:2],scale)
    ytr=train[:,2].astype(np.float64);yv=val[:,2].astype(np.float64);assert_bounds(Ltr,Utr,ytr,'train');assert_bounds(Lv,Uv,yv,'validation')
    lb_val=float(100*np.mean((yv-Lv)/yv));gap_val=float(100*np.mean((Uv-Lv)/yv));print('R4I_INDEX_VALIDATION',lb_val,gap_val,flush=True)
    runs=[];states=[]
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
            if step%500==0:print('R4I_PROGRESS',seed,step,best,best_step,flush=True)
        if best_state is None:raise RuntimeError('no validation checkpoint')
        runs.append({'seed':seed,'best_validation_mre_percent':best,'best_step':best_step,'history':hist,'train_seconds':time.perf_counter()-started});states.append(best_state)
    # Development test is accessed only after all seed checkpoint choices are frozen.
    test=np.load(r4e.DATA)['test'].copy();Xt,Lt,Ut=r4e.pair_features(node,coords,test[:,:2],scale);yt=test[:,2].astype(np.float64);assert_bounds(Lt,Ut,yt,'test');reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);vals=[]
    outdir=RESULT/'coordfps32_residual_mlp_5000';outdir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=r4e.AlphaMLP(Xtr.shape[1]).cuda();m.load_state_dict(state);pred,a=predict(m,Xt,Lt,Ut);met=r4e.metrics(pred,yt,short,reverse);row.update({'test':met,'alpha_mean':float(a.mean()),'alpha_std':float(a.std())});vals.append(met['mre_percent'])
        p=outdir/f"seed{row['seed']}.pt";torch.save({'state_dict':state,'landmarks':landmarks,'scale':scale,'input_dim':Xtr.shape[1]},p);row['checkpoint']=str(p.relative_to(ROOT))
    rec={'status':'completed','classification':'development-only coord-FPS landmark follow-up selected from R4H validation diagnostic; not fresh confirmation',
         'selection_basis':'R4H validation-only: coord_fps32 lower-bound MRE was better than random32 before this residual run',
         'updates':UPDATES,'seeds':list(SEEDS),'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'feature_dim':int(Xtr.shape[1]),'index_build_seconds':build_seconds,
         'validation_lb_mre_percent':lb_val,'validation_relative_interval_gap_mean_percent':gap_val,'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1))}
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4I_COMPLETE',json.dumps({'mean':rec['test_mre_mean'],'sd':rec['test_mre_sd'],'best_val':[r['best_validation_mre_percent'] for r in runs],'steps':[r['best_step'] for r in runs]}),flush=True)
if __name__=='__main__':main()

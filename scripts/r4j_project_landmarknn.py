"""R4J: zero-training projection diagnostic for the directed LandmarkNN baseline.

For every query, the directed landmark index certifies L <= d <= U.  Euclidean projection of any scalar
prediction g onto [L,U] cannot increase |g-d|, hence cannot increase relative absolute error either.
This script applies that post-hoc operator to the already-frozen R4F LandmarkNN checkpoints and to the
unweighted three-seed ensemble.  No training, hyperparameter fitting, or test-driven selection occurs.
The current R4E test is already a development benchmark, not a fresh confirmation set.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
import scripts.r4f_directed_strong_baselines as r4f
from scripts.r4e_build_directed_jinan_workload import label_pairs

R4F=ROOT/'reports/audit-r4f-20260914/directed_landmarknn_alt32.json'
OUTDIR=ROOT/'reports/audit-r4j-20260915';OUT=OUTDIR/'landmarknn_projection.json'


def mre(p,y):return float(100*np.mean(np.abs(np.asarray(p)-y)/y))
def project(p,L,U):return np.minimum(U,np.maximum(L,np.asarray(p,dtype=np.float64)))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    rec=json.load(open(R4F));z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();test=z['test'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25))
    node,landmarks,A,_=r4e.build_index(z)
    if list(map(int,landmarks))!=list(map(int,rec['landmarks'])):raise AssertionError('landmark mismatch')
    # Certified interval endpoints and LandmarkNN features for validation/test.
    _,Lv,Uv=r4e.pair_features(node,coords,val[:,:2],scale);_,Lt,Ut=r4e.pair_features(node,coords,test[:,:2],scale)
    yv=val[:,2].astype(np.float64);yt=test[:,2].astype(np.float64)
    if np.any(Lv>yv+1e-9) or np.any(Uv<yv-1e-9) or np.any(Lt>yt+1e-9) or np.any(Ut<yt-1e-9):raise AssertionError('certified interval violation')
    b,_,l2=r4f.encode_numpy(node,coords,val[:,:2],True);Xv=np.concatenate((b,l2),1)
    b,_,l2=r4f.encode_numpy(node,coords,test[:,:2],True);Xt=np.concatenate((b,l2),1)
    maxd=float(np.max(train[:,2]));device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];pv_all=[];pt_all=[]
    reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1])
    for row in rec['runs']:
        ck=ROOT/row['checkpoint'];obj=torch.load(ck,map_location='cpu',weights_only=False);m=r4f.LandmarkNN(Xv.shape[1],maxd).to(device);m.load_state_dict(obj['state_dict'])
        pv=r4f.predict_nn(m,Xv,device).astype(np.float64);pt=r4f.predict_nn(m,Xt,device).astype(np.float64);pv_all.append(pv);pt_all.append(pt)
        qv=project(pv,Lv,Uv);qt=project(pt,Lt,Ut)
        rawv=mre(pv,yv);projv=mre(qv,yv);rawt=r4e.metrics(pt,yt,short,reverse);projt=r4e.metrics(qt,yt,short,reverse)
        if projv>rawv+1e-10 or projt['mre_percent']>rawt['mre_percent']+1e-10:raise AssertionError(('projection worsened',row['seed']))
        rows.append({'seed':row['seed'],'raw_validation_mre_percent':rawv,'projected_validation_mre_percent':projv,
                     'raw_test':rawt,'projected_test':projt,'validation_projection_changed_fraction':float(np.mean(np.abs(qv-pv)>0)),
                     'test_projection_changed_fraction':float(np.mean(np.abs(qt-pt)>0))})
    pv=np.mean(np.stack(pv_all),axis=0);pt=np.mean(np.stack(pt_all),axis=0);qv=project(pv,Lv,Uv);qt=project(pt,Lt,Ut)
    ensemble={'raw_validation_mre_percent':mre(pv,yv),'projected_validation_mre_percent':mre(qv,yv),
              'raw_test':r4e.metrics(pt,yt,short,reverse),'projected_test':r4e.metrics(qt,yt,short,reverse),
              'validation_projection_changed_fraction':float(np.mean(np.abs(qv-pv)>0)),'test_projection_changed_fraction':float(np.mean(np.abs(qt-pt)>0))}
    if ensemble['projected_validation_mre_percent']>ensemble['raw_validation_mre_percent']+1e-10 or ensemble['projected_test']['mre_percent']>ensemble['raw_test']['mre_percent']+1e-10:raise AssertionError('ensemble projection worsened')
    best_idx=int(np.argmin([x['best_validation_mre_percent'] for x in rec['runs']));best_seed=rec['runs'][best_idx]['seed']
    out={'status':'completed','classification':'zero-training development diagnostic on previously exposed test','lemma':'If L<=d<=U, scalar projection Pi_[L,U](g) cannot increase absolute or relative absolute error.',
         'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'rows':rows,'ensemble3':ensemble,'r4f_validation_best_seed':best_seed}
    OUTDIR.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2)+'\n')
    print('R4J_COMPLETE',json.dumps({'seeds':[(x['seed'],x['raw_test']['mre_percent'],x['projected_test']['mre_percent']) for x in rows],
          'ensemble_raw':ensemble['raw_test']['mre_percent'],'ensemble_projected':ensemble['projected_test']['mre_percent'],'best_seed_by_r4f_validation':best_seed}),flush=True)
if __name__=='__main__':main()

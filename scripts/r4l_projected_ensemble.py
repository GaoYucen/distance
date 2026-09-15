"""R4L: validation-only selection among predeclared LandmarkNN seed ensembles, followed by certified projection.

The three R4F LandmarkNN checkpoints are already frozen.  Enumerate all seven non-empty seed subsets;
for each subset average raw predictions, then project to the certified random32 [L,U] interval.  Choose
the subset with lowest validation MRE, freeze it, and only then report its already-exposed development
test result.  This measures the attainable ensemble reference; model-size multiplication is reported and
this is not a fresh confirmation experiment.
"""
from __future__ import annotations
import itertools,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e
import scripts.r4f_directed_strong_baselines as r4f
from scripts.r4e_build_directed_jinan_workload import label_pairs

R4F=ROOT/'reports/audit-r4f-20260914/directed_landmarknn_alt32.json';OUTDIR=ROOT/'reports/audit-r4l-20260915';OUT=OUTDIR/'projected_ensemble.json'
def project(p,L,U):return np.minimum(U,np.maximum(L,np.asarray(p,dtype=np.float64)))
def mre(p,y):return float(100*np.mean(np.abs(np.asarray(p)-y)/y))

def main():
    if OUT.exists():raise FileExistsError(OUT)
    rec=json.load(open(R4F));z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();test=z['test'].copy();coords=z['coordinates'];scale=float(train[:,2].mean());short=float(np.quantile(train[:,2],.25));node,landmarks,A,_=r4e.build_index(z)
    if list(map(int,landmarks))!=list(map(int,rec['landmarks'])):raise AssertionError('landmark mismatch')
    _,Lv,Uv=r4e.pair_features(node,coords,val[:,:2],scale);_,Lt,Ut=r4e.pair_features(node,coords,test[:,:2],scale);yv=val[:,2].astype(np.float64);yt=test[:,2].astype(np.float64)
    b,_,l2=r4f.encode_numpy(node,coords,val[:,:2],True);Xv=np.concatenate((b,l2),1);b,_,l2=r4f.encode_numpy(node,coords,test[:,:2],True);Xt=np.concatenate((b,l2),1)
    device='cuda' if torch.cuda.is_available() else 'cpu';maxd=float(np.max(train[:,2]));seeds=[int(r['seed']) for r in rec['runs']];pv={};pt={};param_count=None
    for row in rec['runs']:
        obj=torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False);m=r4f.LandmarkNN(Xv.shape[1],maxd).to(device);m.load_state_dict(obj['state_dict']);param_count=sum(p.numel() for p in m.parameters())
        pv[int(row['seed'])]=r4f.predict_nn(m,Xv,device).astype(np.float64);pt[int(row['seed'])]=r4f.predict_nn(m,Xt,device).astype(np.float64)
    candidates=[]
    for k in range(1,len(seeds)+1):
        for subset in itertools.combinations(seeds,k):
            av=np.mean(np.stack([pv[s] for s in subset]),axis=0);qv=project(av,Lv,Uv);candidates.append({'seeds':list(subset),'k':k,'validation_raw_mre_percent':mre(av,yv),'validation_projected_mre_percent':mre(qv,yv)})
    candidates.sort(key=lambda x:(x['validation_projected_mre_percent'],x['k'],x['seeds']));chosen=candidates[0];subset=chosen['seeds']
    at=np.mean(np.stack([pt[s] for s in subset]),axis=0);qt=project(at,Lt,Ut);reverse=label_pairs(A,test[:,:2].astype(np.int64)[:,::-1]);raw=r4e.metrics(at,yt,short,reverse);proj=r4e.metrics(qt,yt,short,reverse)
    if proj['mre_percent']>raw['mre_percent']+1e-10:raise AssertionError('projection worsened')
    out={'status':'completed','classification':'development-only validation-selected ensemble reference; current test already exposed','selection_rule':'minimum projected validation MRE among all 7 non-empty subsets of the three frozen R4F seeds; ties prefer fewer models then lexicographic seeds',
         'candidates':candidates,'chosen':chosen,'test_raw':raw,'test_projected':proj,'single_model_parameter_count':param_count,'ensemble_parameter_count':int(param_count*len(subset)),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'test_projection_changed_fraction':float(np.mean(np.abs(qt-at)>0))}
    OUTDIR.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2)+'\n');print('R4L_COMPLETE',json.dumps({'chosen':chosen,'test_raw':raw['mre_percent'],'test_projected':proj['mre_percent'],'short':proj['short_mre_percent'],'asym':proj.get('high_asymmetry_mre_percent')}),flush=True)
if __name__=='__main__':main()

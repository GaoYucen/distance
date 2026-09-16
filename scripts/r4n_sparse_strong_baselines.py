"""Run the already-authorized ALT32 / Dir-CatBoost / Dir-LandmarkNN baselines from a frozen 64-scalar directed-landmark table.

This is the sparse-graph data-path equivalent of r4n_matrix_strong_baselines_v3.py. No model family,
feature recipe, seed, validation rule, or time budget is changed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import scripts.r4n_matrix_strong_baselines_v3 as v3
base=v3.base

SEEDS_FINAL=(42,99,1234); LANDMARK_SEED=20260914

def load_common(workload,index):
    z=np.load(workload); idx=np.load(index)
    train=z['train'].copy(); val=z['validation'].copy(); coords=z['coordinates'].astype(np.float32); n=len(coords)
    node=idx['features'].astype(np.float32); landmarks=idx['landmarks'].astype(np.int64)
    if node.shape!=(n,64) or landmarks.shape!=(32,): raise RuntimeError((node.shape,landmarks.shape,n))
    train_nodes=np.unique(train[:,:2].astype(np.int64)); rng=np.random.default_rng(LANDMARK_SEED); expected=np.sort(rng.choice(train_nodes,size=32,replace=False))
    if not np.array_equal(landmarks,expected): raise RuntimeError('landmarks differ from frozen protocol')
    short=float(np.quantile(train[:,2],.25))
    return z,train,val,coords,node,landmarks,short

def reverse_test(z):
    if 'test_reverse_distances' not in z: raise RuntimeError('missing exact reverse test distances')
    return z['test_reverse_distances'].astype(np.float64)

def eval_alt(z,node,short):
    test=z['test'].copy(); y=test[:,2].astype(np.float64); lo,hi=base.directed_bounds(node,test[:,:2])
    if np.any(lo>y+1e-5) or np.any(hi<y-1e-5): raise AssertionError('directed landmark certificate violation')
    return {'test':base.metric(lo,y,reverse_test(z),short)}

def eval_cat(model,rec,z,coords,node,short,path):
    test=z['test'].copy(); b,l1,_=base.encode(node,coords,test[:,:2],False); X=np.concatenate((b,l1),1); y=test[:,2].astype(np.float64)
    pred=model.predict(X); path.parent.mkdir(parents=True,exist_ok=True); model.save_model(path); out=dict(rec); out['test']=base.metric(pred,y,reverse_test(z),short); out['model_path']=str(path); return out

def eval_lnn(d,maxd,runs,states,z,coords,node,short,outdir):
    test=z['test'].copy(); b,_,l2=base.encode(node,coords,test[:,:2],True); X=np.concatenate((b,l2),1); y=test[:,2].astype(np.float64); vals=[]; outdir.mkdir(parents=True,exist_ok=True)
    for row,state in zip(runs,states):
        m=base.LandmarkNN(d,maxd).cuda(); m.load_state_dict(state); pred=base.predict_nn(m,X); row['test']=base.metric(pred,y,reverse_test(z),short); vals.append(row['test']['mre_percent']); p=outdir/f"seed{row['seed']}.pt"; torch.save({'state_dict':state,'input_dim':d,'max_distance':maxd},p); row['checkpoint']=str(p)
    return {'feature_dim':int(d),'runs':runs,'test_mre_mean':float(np.mean(vals)),'test_mre_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--workload',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--model-dir',type=Path,required=True); ap.add_argument('--models',nargs='+',choices=['alt32','catboost','landmarknn'],default=['alt32','catboost','landmarknn']); ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS_FINAL)); ap.add_argument('--time-limit-seconds',type=float,default=300.0); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    z,train,val,coords,node,landmarks,short=load_common(args.workload,args.index); seeds=tuple(args.seeds); final=(seeds==SEEDS_FINAL and float(args.time_limit_seconds)==300.0)
    rec={'status':'completed','dataset':args.dataset,'classification':'final directed-feature baseline evaluation from frozen sparse landmark index' if final else 'bounded screening','workload_sha256':base.sha256(args.workload),'index_sha256':base.sha256(args.index),'node_count':int(len(coords)),'train_rows':int(len(train)),'validation_rows':int(len(val)),'test_rows':int(len(z['test'])),'landmark_seed':LANDMARK_SEED,'landmarks':landmarks.tolist(),'index_scalars_per_node':64,'index_bytes_per_node_float32':256,'requested_seeds':list(seeds),'time_limit_seconds_per_model_seed':float(args.time_limit_seconds),'models':{}}
    if 'alt32' in args.models: rec['models']['ALT32']=eval_alt(z,node,short)
    if 'catboost' in args.models:
        m,r=base.train_catboost(train,val,coords,node,args.time_limit_seconds); rec['models']['Dir-CatBoost']=eval_cat(m,r,z,coords,node,short,args.model_dir/'dir_catboost.cbm')
    if 'landmarknn' in args.models:
        d,maxd,runs,states=base.train_landmarknn(train,val,coords,node,args.time_limit_seconds,seeds); rec['models']['Dir-LandmarkNN']=eval_lnn(d,maxd,runs,states,z,coords,node,short,args.model_dir/'dir_landmarknn')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rec,indent=2)+'\n'); print('R4N_SPARSE_STRONG_BASELINES_COMPLETE',json.dumps({'dataset':args.dataset,'summary':{k:(v.get('test',{}).get('mre_percent') if 'test' in v else v.get('test_mre_mean')) for k,v in rec['models'].items()}}),flush=True)
if __name__=='__main__': main()

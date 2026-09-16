"""R4H: validation-only landmark-selection diagnostic at a fixed 32 directed landmarks.

No test labels are loaded.  This diagnostic asks whether the current random ALT32 index is leaving
large, easily recoverable validation accuracy on the table.  It is not a final model comparison and
standard landmark heuristics are not claimed as novel.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4e_directed_landmark_residual as r4e

REPORT=ROOT/'reports/audit-r4h-20260914';OUT=REPORT/'landmark_selection_validation.json'
K=32


def random_landmarks(train_nodes):
    rng=np.random.default_rng(20260914);return np.sort(rng.choice(train_nodes,size=K,replace=False))

def frequency_order(train,train_nodes):
    n=int(train_nodes.max())+1;cnt=np.bincount(train[:,:2].astype(np.int64).ravel(),minlength=n);order=train_nodes[np.argsort(-cnt[train_nodes],kind='stable')];return order

def fps_order(coords,train_nodes):
    xy=np.asarray(coords,dtype=np.float64)[train_nodes];mu=xy.mean(0);sd=xy.std(0);sd=np.where(sd<1e-12,1.,sd);x=(xy-mu)/sd
    start=int(np.argmin(np.square(x).sum(1)));picked=[start];mind=np.square(x-x[start]).sum(1);mind[start]=-1
    while len(picked)<K:
        j=int(np.argmax(mind));picked.append(j);mind=np.minimum(mind,np.square(x-x[j]).sum(1));mind[picked]=-1
    return train_nodes[np.array(picked,dtype=np.int64)]

def hybrid(coords,train,train_nodes):
    fps=list(map(int,fps_order(coords,train_nodes)[:16]));freq=list(map(int,frequency_order(train,train_nodes)));out=fps.copy()
    for v in freq:
        if v not in out:out.append(v)
        if len(out)==K:break
    return np.array(out,dtype=np.int64)

def certified_bounds(A,landmarks,pairs):
    fr=dijkstra(A,directed=True,indices=landmarks);to=dijkstra(A.T,directed=True,indices=landmarks)
    # validation diagnostic is evaluated in float64; deployment float32 certification is separately handled by R4E.
    u,v=np.asarray(pairs,dtype=np.int64).T
    low=np.maximum(fr[:,v]-fr[:,u],to[:,u]-to[:,v]).max(0);L=np.maximum(0.,low)
    U=(to[:,u]+fr[:,v]).min(0);return L,U

def main():
    if OUT.exists():raise FileExistsError(OUT)
    z=np.load(r4e.DATA);train=z['train'].copy();val=z['validation'].copy();coords=z['coordinates'];A=r4e.r4c.build_native_csr('Jinan_native_directed',z,r4e.RAW);train_nodes=np.unique(train[:,:2].astype(np.int64));y=val[:,2].astype(np.float64)
    strategies={'random32':random_landmarks(train_nodes),'coord_fps32':fps_order(coords,train_nodes),'frequency32':frequency_order(train,train_nodes)[:K],'hybrid16fps16freq':hybrid(coords,train,train_nodes)}
    rows={}
    for name,lms in strategies.items():
        st=time.perf_counter();L,U=certified_bounds(A,lms,val[:,:2]);rel=(y-L)/y;gap=(U-L)/y
        if np.any(L>y+1e-8) or np.any(U<y-1e-8):raise AssertionError(name)
        rows[name]={'landmarks':list(map(int,lms)),'validation_lb_mre_percent':float(100*np.mean(rel)),'validation_lb_p95_percent':float(100*np.quantile(rel,.95)),
                    'validation_relative_interval_gap_mean_percent':float(100*np.mean(gap)),'validation_relative_interval_gap_median_percent':float(100*np.median(gap)),'build_eval_seconds':time.perf_counter()-st}
        print('R4H',name,rows[name]['validation_lb_mre_percent'],rows[name]['validation_relative_interval_gap_mean_percent'],flush=True)
    REPORT.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps({'status':'completed','classification':'validation-only landmark diagnostic; no test access','K':K,'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()

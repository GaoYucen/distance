"""Build a fresh 500K uniform ordered-OD confirmation workload on native-directed Shenzhen.
The workload and exact all-pairs matrix are frozen before the frozen R4M model is run.
"""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4c_realroads as r4c
BASE=ROOT/'data/protocol_r2/Shenzhen_native_directed_uniform.npz'
BASE_SHA='8cede9cc3b77d22acf2877ee85e0d6fa34d05785bf05b1b4cc9a3f86566fc92c'
RAW=ROOT/'data/figshare_native_20260913/edge_shenzhen.csv'
MATRIX=ROOT/'data/audit/Shenzhen_native_directed_allpairs_r4n.npy'
OUT=ROOT/'data/protocol_r4n/Shenzhen_native_directed_uniform500k_fresh.npz'
META=ROOT/'reports/audit-r4n-20260915/shenzhen_fresh_workload.json'
SEED=2026091503

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def sample_pairs(n,m,rng):
    k=rng.choice(n*(n-1),size=m,replace=False);u=k//(n-1);r=k%(n-1);v=r+(r>=u);return np.column_stack((u,v)).astype(np.int64)

def main():
    if sha(BASE)!=BASE_SHA:raise RuntimeError('base Shenzhen protocol hash mismatch')
    z=np.load(BASE);coords=z['coordinates'].astype(np.float32);n=len(coords);A=r4c.build_native_csr('Shenzhen_native_directed_uniform',z,RAW)
    started=time.perf_counter()
    if MATRIX.exists():D=np.load(MATRIX,mmap_mode='r')
    else:
        D=dijkstra(A,directed=True)
        if D.shape!=(n,n) or not np.isfinite(D).all():raise RuntimeError('nonfinite Shenzhen all-pairs matrix')
        MATRIX.parent.mkdir(parents=True,exist_ok=True);np.save(MATRIX,D);del D;D=np.load(MATRIX,mmap_mode='r')
    build_seconds=time.perf_counter()-started
    if D.shape!=(n,n):raise AssertionError(D.shape)
    if OUT.exists() or META.exists():raise FileExistsError('fresh Shenzhen workload already frozen')
    rng=np.random.default_rng(SEED);pairs=sample_pairs(n,500000,rng);dist=np.asarray(D[pairs[:,0],pairs[:,1]],dtype=np.float64)
    if np.any(dist<=0) or not np.isfinite(dist).all():raise RuntimeError('invalid labels')
    allq=np.column_stack((pairs,dist));train=allq[:400000];val=allq[400000:450000];test=allq[450000:]
    keys=[set(map(tuple,x[:,:2].astype(np.int64))) for x in (train,val,test)];ov=[len(keys[0]&keys[1]),len(keys[0]&keys[2]),len(keys[1]&keys[2])]
    if any(ov):raise AssertionError(ov)
    OUT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT,train=train,validation=val,test=test,coordinates=coords,original_node_ids=z['original_node_ids'].astype(np.int64))
    # independent sparse-Dijkstra check on a deterministic subset
    rr=np.random.default_rng(919);ix=rr.choice(len(test),size=128,replace=False);chk=test[ix];src=np.unique(chk[:,0].astype(np.int64));Ds=dijkstra(A,directed=True,indices=src);pos={int(s):i for i,s in enumerate(src)};err=max(abs(float(Ds[pos[int(u)],int(v)])-float(d)) for u,v,d in chk)
    if err>1e-7:raise AssertionError(err)
    rev=np.asarray(D[test[:,1].astype(np.int64),test[:,0].astype(np.int64)],dtype=np.float64);alpha=np.abs(test[:,2]-rev)/((test[:,2]+rev)/2)
    rec={'status':'frozen','classification':'fresh confirmation dataset; workload frozen before R4M evaluation','seed':SEED,'node_count':n,'splits':[len(train),len(val),len(test)],'ordered_pair_overlaps':ov,'base_sha256':BASE_SHA,'raw_edge_sha256':sha(RAW),'matrix_sha256':sha(MATRIX),'workload_sha256':sha(OUT),'allpairs_build_or_load_seconds':build_seconds,'independent_check_max_abs_m':float(err),'test_alpha_ge_20pct_fraction':float(np.mean(alpha>=.2)),'test_mean_alpha_percent':float(100*alpha.mean())}
    META.parent.mkdir(parents=True,exist_ok=True);META.write_text(json.dumps(rec,indent=2)+'\n');print('R4N_SHENZHEN_FRESH',json.dumps(rec),flush=True)
if __name__=='__main__':main()

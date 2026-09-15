"""Freeze a 500K ordered-OD workload on the legacy native-directed Chengdu graph.
This script runs before any R4M evaluation on this workload.
"""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'data/audit/chengdu_directed_shortest_distance_matrix.npy'
NODES=ROOT/'data/audit/chengdu_node-mod.txt'
OUT=ROOT/'data/protocol_r4n/Chengdu_native_directed_uniform500k.npz'
META=ROOT/'reports/audit-r4n-20260915/chengdu_workload.json'
SEED=2026091502

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def sample_pairs(n,m,rng):
    k=rng.choice(n*(n-1),size=m,replace=False);u=k//(n-1);r=k%(n-1);v=r+(r>=u);return np.column_stack((u,v)).astype(np.int64)

def main():
    if OUT.exists() or META.exists():raise FileExistsError('Chengdu R4N workload already frozen')
    D=np.load(MATRIX,mmap_mode='r');n=D.shape[0]
    if D.shape!=(n,n) or not np.isfinite(D).all():raise RuntimeError('bad matrix')
    off=np.asarray(D[np.arange(n)[:,None],np.arange(n)[None,:]]) if False else None
    with open(NODES,newline='') as f:
        rows=list(csv.reader(f)); header=rows[0]; vals=rows[1:]
    coords=np.asarray([[float(r[1]),float(r[2])] for r in vals],dtype=np.float32)
    if len(coords)!=n:raise AssertionError((len(coords),n,header))
    rng=np.random.default_rng(SEED);pairs=sample_pairs(n,500000,rng);d=np.asarray(D[pairs[:,0],pairs[:,1]],dtype=np.float64)
    if np.any(d<=0) or not np.isfinite(d).all():raise RuntimeError('invalid nonself distances')
    allq=np.column_stack((pairs,d));train=allq[:400000];val=allq[400000:450000];test=allq[450000:]
    keys=[set(map(tuple,x[:,:2].astype(np.int64))) for x in (train,val,test)];ov=[len(keys[0]&keys[1]),len(keys[0]&keys[2]),len(keys[1]&keys[2])]
    if any(ov):raise AssertionError(ov)
    OUT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT,train=train,validation=val,test=test,coordinates=coords,original_node_ids=np.arange(n,dtype=np.int64))
    rev=np.asarray(D[test[:,1].astype(np.int64),test[:,0].astype(np.int64)],dtype=np.float64);alpha=np.abs(test[:,2]-rev)/((test[:,2]+rev)/2)
    rec={'status':'frozen','classification':'additional native-directed dataset; uniform ordered OD; not used to tune R4M architecture','seed':SEED,'node_count':n,'splits':[len(train),len(val),len(test)],'ordered_pair_overlaps':ov,'matrix_sha256':sha(MATRIX),'nodes_sha256':sha(NODES),'output_sha256':sha(OUT),'test_alpha_ge_20pct_fraction':float(np.mean(alpha>=.2)),'test_mean_alpha_percent':float(100*alpha.mean())}
    META.parent.mkdir(parents=True,exist_ok=True);META.write_text(json.dumps(rec,indent=2)+'\n');print('R4N_CHENGDU_WORKLOAD',json.dumps(rec),flush=True)
if __name__=='__main__':main()

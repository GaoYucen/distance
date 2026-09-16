"""Build the survey-matched Jinan workload with native directed shortest-path labels.

The source OD rows and split membership come from the pinned Purdue survey repo.
The stored undirected distance labels are intentionally ignored.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.r4c_realroads as r4c

SURVEY=Path('/workspace/shortest-distance-survey-r4e')
SURVEY_PIN='dcaa89d38300bfb823eda84ccdfc85c42edbeae8'
BASE=ROOT/'data/protocol_r2/Jinan_native_directed.npz'
RAW=ROOT/'data/figshare_native_20260913/edge_jinan.csv'
OUT=ROOT/'data/protocol_r4e/Jinan_native_directed_workload_500k.npz'
REPORT=ROOT/'reports/audit-r4e-20260914/directed_workload.json'
SPLITS={'train':'W_Jinan_train.queries.npz','validation':'W_Jinan_val.queries.npz','test':'W_Jinan_test.queries.npz'}


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def label_pairs(A, ids, batch_sources=64):
    ids=np.asarray(ids,dtype=np.int64);u,v=ids.T
    out=np.full(len(ids),np.inf,dtype=np.float64)
    order=np.argsort(u,kind='stable');su=u[order]
    unique=np.unique(u)
    for begin in range(0,len(unique),batch_sources):
        sources=unique[begin:begin+batch_sources]
        D=dijkstra(A,directed=True,indices=sources)
        for row,s in enumerate(sources):
            lo=np.searchsorted(su,s,'left');hi=np.searchsorted(su,s,'right');ix=order[lo:hi]
            out[ix]=D[row,v[ix]]
    if not np.isfinite(out).all():raise RuntimeError('nonfinite directed label inside frozen SCC')
    return out


def main():
    if OUT.exists() or REPORT.exists():raise FileExistsError('R4E directed workload artifacts already exist')
    if not (SURVEY/'.git').exists():raise FileNotFoundError(SURVEY)
    import subprocess
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SURVEY,text=True).strip()
    if head!=SURVEY_PIN:raise RuntimeError((head,SURVEY_PIN))
    z=np.load(BASE);original=z['original_node_ids'].astype(np.int64);n=len(original)
    mapping=np.full(int(original.max())+1,-1,dtype=np.int64);mapping[original]=np.arange(n)
    A=r4c.build_native_csr('Jinan_native_directed',z,RAW)
    if A.shape!=(n,n):raise AssertionError(A.shape)
    source_dir=SURVEY/'data/W_Jinan/real_workload_perturb_500k'
    arrays={};info={};keys={};started=time.perf_counter()
    for split,name in SPLITS.items():
        p=source_dir/name;q=np.load(p,allow_pickle=False)
        src=q['src'].astype(np.int64)-1;dst=q['dst'].astype(np.int64)-1
        if src.min()<0 or dst.min()<0:raise AssertionError('survey ids must be 1-based positive')
        inmap=(src<len(mapping))&(dst<len(mapping))
        mapped=np.full((len(src),2),-1,dtype=np.int64)
        mapped[inmap,0]=mapping[src[inmap]];mapped[inmap,1]=mapping[dst[inmap]]
        valid=inmap&(mapped[:,0]>=0)&(mapped[:,1]>=0)&(mapped[:,0]!=mapped[:,1])
        ids=mapped[valid]
        labels=label_pairs(A,ids)
        arr=np.column_stack((ids,labels))
        if np.any(arr[:,2]<=0):raise AssertionError('nonpositive nonself directed label')
        arrays[split]=arr
        k=ids[:,0].astype(np.int64)*n+ids[:,1].astype(np.int64);keys[split]=np.unique(k)
        info[split]={'source_rows':int(len(src)),'retained_rows':int(len(arr)),'excluded_outside_scc_or_self':int((~valid).sum()),
                     'unique_directed_pairs':int(len(np.unique(k))),'source_sha256':sha(p)}
        print('R4E_DIRECTED_SPLIT',split,info[split],flush=True)
    overlaps={f'{a}/{b}':int(len(np.intersect1d(keys[a],keys[b]))) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    # Independent NetworkX checks on 128 deterministic rows pooled across splits.
    G=nx.from_scipy_sparse_array(A,create_using=nx.DiGraph)
    pool=np.vstack([arrays[k] for k in ['train','validation','test']]);rng=np.random.default_rng(20260914)
    ix=rng.choice(len(pool),size=min(128,len(pool)),replace=False);err=[]
    for row in pool[ix]:
        u,v,d=int(row[0]),int(row[1]),float(row[2]);d2=float(nx.shortest_path_length(G,u,v,weight='weight'));err.append(abs(d-d2))
    maxerr=float(max(err,default=0.))
    if maxerr>1e-7:raise AssertionError(maxerr)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT,train=arrays['train'],validation=arrays['validation'],test=arrays['test'],
                        coordinates=z['coordinates'],edges=z['edges'],original_node_ids=original)
    report={'status':'passed','classification':'survey-matched directed development workload; split rows preserved where inside SCC',
            'survey_commit':SURVEY_PIN,'base_protocol_sha256':sha(BASE),'raw_edge_sha256':sha(RAW),'output':str(OUT.relative_to(ROOT)),
            'output_sha256':sha(OUT),'node_count':n,'split_info':info,'directed_pair_overlap':overlaps,
            'networkx_check':{'directions':len(err),'max_abs_difference_m':maxerr,'passed':True},
            'runtime_seconds':time.perf_counter()-started,
            'warning':'Stored survey undirected dist labels were not used. Existing split duplicate OD rows are intentionally preserved for benchmark mirroring.'}
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('R4E_DIRECTED_WORKLOAD_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()

"""Export Survey-compatible graph/query assets for the frozen native-directed FLA workload.

For Vdist2vec/ANEDA/RNE directed-target adaptations only. Native-directed shortest-path labels and the
pre-frozen 400K/50K/50K OD split are preserved verbatim. Graph-dependent Survey preprocessing remains
undirected by collapsing native arcs to unordered pairs with minimum observed weight.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--graph',type=Path,required=True); ap.add_argument('--workload',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--prefix',default='R4N_FLA'); args=ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    g=np.load(args.graph); z=np.load(args.workload); coords=np.asarray(g['coordinates'],dtype=np.float64); n=len(coords)
    if len(z['coordinates'])!=n: raise RuntimeError('graph/workload node count mismatch')
    src=np.asarray(g['src'],dtype=np.int64); dst=np.asarray(g['dst'],dtype=np.int64); wt=np.asarray(g['weight'],dtype=np.float64)
    a=np.minimum(src,dst); b=np.maximum(src,dst); mask=a!=b; a,b,wt=a[mask],b[mask],wt[mask]
    key=a*np.int64(n)+b; order=np.argsort(key,kind='stable'); key=key[order]; a=a[order]; b=b[order]; wt=wt[order]
    starts=np.r_[0, np.flatnonzero(key[1:]!=key[:-1])+1]; minw=np.minimum.reduceat(wt,starts); ua=a[starts]; ub=b[starts]
    args.output_dir.mkdir(parents=True)
    edge=args.output_dir/f'{args.prefix}.edges'; nodes=args.output_dir/f'{args.prefix}.nodes'
    with edge.open('w',newline='') as f:
        w=csv.writer(f)
        for u,v,d in zip(ua,ub,minw): w.writerow((int(u)+1,int(v)+1,float(d)))
    with nodes.open('w',newline='') as f:
        w=csv.writer(f)
        for i,(x,y) in enumerate(coords): w.writerow((i+1,float(x),float(y)))
    qdir=args.output_dir/'directed_target_500k'; qdir.mkdir(); files={}
    for k,suf in (('train','train'),('validation','val'),('test','test')):
        q=np.asarray(z[k]); p=qdir/f'{args.prefix}_{suf}.queries.npz'; np.savez_compressed(p,src=q[:,0].astype(np.int64)+1,dst=q[:,1].astype(np.int64)+1,dist=q[:,2].astype(np.float32)); files[k]={'rows':int(len(q)),'sha256':sha256(p)}
    meta={'status':'completed','classification':'Survey-style directed-target adaptation asset; exact native-directed targets with original undirected graph-side inductive bias','node_count':int(n),'native_directed_arc_count':int(len(src)),'survey_undirected_edge_count':int(len(ua)),'undirected_collapse_rule':'unordered pair minimum weight over native directed arcs','graph_sha256':sha256(args.graph),'workload_sha256':sha256(args.workload),'edges_sha256':sha256(edge),'nodes_sha256':sha256(nodes),'query_files':files,'directional_capacity_added':False}
    (args.output_dir/'DIRECTED_TARGET_ASSET_META.json').write_text(json.dumps(meta,indent=2)+'\n'); print('R4N_SURVEY_FLA_ASSETS',json.dumps(meta))
if __name__=='__main__': main()

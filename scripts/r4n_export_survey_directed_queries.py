"""Export the frozen native-directed workload into the survey repo query-file format.

This does not alter graph preprocessing or model code.  It replaces only OD distance labels/splits so
survey baselines can be evaluated on exactly the same directed target workload used by R4M.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/protocol_r4e/Jinan_native_directed_workload_500k.npz'
EXPECTED='384e22b76692b5a409131fccd8a36bb02f879c25c160766d4a407142a9cf8c68'

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(path:Path,a):
    a=np.asarray(a)
    src=a[:,0].astype(np.int64)+1;dst=a[:,1].astype(np.int64)+1;dist=a[:,2].astype(np.float32)
    np.savez_compressed(path,src=src,dst=dst,dist=dist)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=DEFAULT);ap.add_argument('--out_dir',type=Path,required=True);ap.add_argument('--prefix',default='W_Jinan');args=ap.parse_args()
    if args.input.resolve()==DEFAULT.resolve() and sha256(args.input)!=EXPECTED:raise RuntimeError('frozen Jinan workload hash mismatch')
    z=np.load(args.input);args.out_dir.mkdir(parents=True,exist_ok=False)
    mapping={'train':'train','validation':'val','test':'test'};meta={}
    for key,suffix in mapping.items():
        a=z[key];p=args.out_dir/f'{args.prefix}_{suffix}.queries.npz';write(p,a);meta[key]={'rows':int(len(a)),'sha256':sha256(p)}
    # verify directed OD disjointness exactly as ordered pairs
    sets={k:set(map(tuple,z[k][:,:2].astype(np.int64))) for k in mapping}
    overlaps={'train_val':len(sets['train']&sets['validation']),'train_test':len(sets['train']&sets['test']),'val_test':len(sets['validation']&sets['test'])}
    if any(overlaps.values()):raise RuntimeError(overlaps)
    (args.out_dir/'DIRECTED_TARGET_META.json').write_text(json.dumps({'source':str(args.input),'source_sha256':sha256(args.input),'classification':'same OD/split, native-directed labels; survey graph-dependent preprocessing may remain undirected unless separately adapted','files':meta,'ordered_pair_overlaps':overlaps},indent=2)+'\n')
    print(json.dumps({'out':str(args.out_dir),'rows':{k:int(len(z[k])) for k in mapping},'overlaps':overlaps},indent=2))
if __name__=='__main__':main()

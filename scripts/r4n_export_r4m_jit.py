"""Export the frozen R4M model as an end-to-end node-ID -> distance TorchScript module.

The exported module includes all online work that must be counted in query latency:
  node/coordinate gathers, directed ALT L/U reductions, normalized pair features,
  the bounded MLP, and final L + alpha*(U-L) decode.
Offline Dijkstra/index construction is not inside forward and is reported separately.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4m_bounded_landmarknn as r4m
import scripts.r4e_directed_landmark_residual as r4e

REPORT=ROOT/'reports/audit-r4m-20260915/bounded_landmarknn.json'
INDEX=ROOT/'results/audit-r4e-20260914/directed_landmark_residual/ALT32_index.npz'
DATA=r4e.DATA

class R4MEndToEnd(nn.Module):
    def __init__(self,node,coords,scale,bound_mu,bound_sd,state):
        super().__init__()
        node=np.asarray(node,dtype=np.float32);coords=np.asarray(coords,dtype=np.float32)
        # Survey-style normalized LandmarkNN features are constants per node and can be cached in the index.
        nmu=node.mean(0,keepdims=True,dtype=np.float64);nsd=node.std(0,keepdims=True,dtype=np.float64);nsd=np.where(nsd<1e-8,1.,nsd)
        cmu=coords.mean(0,keepdims=True,dtype=np.float64);csd=coords.std(0,keepdims=True,dtype=np.float64);csd=np.where(csd<1e-8,1.,csd)
        node_norm=((node.astype(np.float64)-nmu)/nsd).astype(np.float32)
        coord_norm=((coords.astype(np.float64)-cmu)/csd).astype(np.float32)
        lo=np.nextafter(node,np.float32(-np.inf),dtype=np.float32)
        hi=np.nextafter(node,np.float32(np.inf),dtype=np.float32)
        self.register_buffer('node_norm',torch.from_numpy(node_norm),persistent=True)
        self.register_buffer('coord_norm',torch.from_numpy(coord_norm),persistent=True)
        self.register_buffer('node_lo',torch.from_numpy(lo),persistent=True)
        self.register_buffer('node_hi',torch.from_numpy(hi),persistent=True)
        self.register_buffer('bound_mu',torch.as_tensor(np.asarray(bound_mu,dtype=np.float32)),persistent=True)
        self.register_buffer('bound_sd',torch.as_tensor(np.asarray(bound_sd,dtype=np.float32)),persistent=True)
        self.scale=float(scale)
        self.fc1=nn.Linear(139,1024);self.fc2=nn.Linear(1024,512);self.fc3=nn.Linear(512,1)
        self.load_state_dict({**self.state_dict(),
                              'fc1.weight':state['fc1.weight'],'fc1.bias':state['fc1.bias'],
                              'fc2.weight':state['fc2.weight'],'fc2.bias':state['fc2.bias'],
                              'fc3.weight':state['fc3.weight'],'fc3.bias':state['fc3.bias']},strict=True)

    def forward(self,src:torch.Tensor,dst:torch.Tensor)->torch.Tensor:
        u=src.long();v=dst.long()
        # Normalized LandmarkNN features.
        a=self.node_norm.index_select(0,u);b=self.node_norm.index_select(0,v)
        ca=self.coord_norm.index_select(0,u);cb=self.coord_norm.index_select(0,v)
        cos=(a*b).sum(1)/torch.clamp(torch.sqrt((a*a).sum(1)*(b*b).sum(1)),min=1e-12)
        l2=torch.sqrt(torch.clamp(((ca-cb)*(ca-cb)).sum(1),min=0.0))
        base=torch.cat((a,b,ca,cb,cos.unsqueeze(1),l2.unsqueeze(1)),1)

        # Certified directed landmark interval from one-ULP outward-rounded float32 index values.
        lo_u=self.node_lo.index_select(0,u);lo_v=self.node_lo.index_select(0,v)
        hi_u=self.node_hi.index_select(0,u);hi_v=self.node_hi.index_select(0,v)
        # first 32 = d(l,node); last 32 = d(node,l)
        lower1=lo_v[:,:32]-hi_u[:,:32]
        lower2=lo_u[:,32:]-hi_v[:,32:]
        L=torch.clamp(torch.maximum(lower1.max(1).values,lower2.max(1).values),min=0.0)
        U=(hi_u[:,32:]+hi_v[:,:32]).min(1).values
        G=torch.clamp(U-L,min=0.0)
        eps=1e-9
        e=torch.stack((torch.log1p(L/self.scale),torch.log1p(U/self.scale),torch.log1p(G/self.scale),
                       L/torch.clamp(U,min=eps),G/torch.clamp(U,min=eps)),1)
        e=(e-self.bound_mu)/self.bound_sd
        x=torch.cat((base,e),1)
        h=F.relu(self.fc1(x));h=F.relu(self.fc2(h));alpha=torch.sigmoid(self.fc3(h)).squeeze(1)
        return L+alpha*G

def numpy_reference(seed,src,dst):
    rec=json.load(open(REPORT));run=next(x for x in rec['runs'] if int(x['seed'])==seed);ck=torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=False)
    iz=np.load(INDEX);node=iz['node_features'];z=np.load(DATA);coords=z['coordinates'];scale=float(z['train'][:,2].mean())
    X,L,U,_,_=r4m.build_features(node,coords,np.column_stack((src,dst)),scale,ck['bound_extra_mu'],ck['bound_extra_sd'])
    m=r4m.BoundedLandmarkNN(X.shape[1],ck['alpha0']);m.load_state_dict(ck['state_dict']);m.eval()
    with torch.no_grad():alpha=m(torch.from_numpy(X)).numpy()
    return L+alpha*(U-L),ck,node,coords,scale

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--output',type=Path,default=ROOT/'results/audit-r4n-20260915/r4m_end_to_end_seed1234.jit.pt');args=ap.parse_args()
    rec=json.load(open(REPORT));val={int(x['seed']):x['best_validation_mre_percent'] for x in rec['runs']};best=min(val,key=val.get)
    if args.seed!=best:print('WARNING selected seed is not validation-best',args.seed,best,flush=True)
    z=np.load(DATA);pairs=z['validation'][:4096,:2].astype(np.int64);src,dst=pairs.T
    ref,ck,node,coords,scale=numpy_reference(args.seed,src,dst)
    mod=R4MEndToEnd(node,coords,scale,ck['bound_extra_mu'],ck['bound_extra_sd'],ck['state_dict']).eval()
    with torch.no_grad():p=mod(torch.from_numpy(src),torch.from_numpy(dst)).numpy()
    max_abs=float(np.max(np.abs(p-ref)));max_rel=float(np.max(np.abs(p-ref)/np.maximum(np.abs(ref),1e-9)))
    if max_abs>0.05 or max_rel>1e-4:raise AssertionError((max_abs,max_rel))
    scripted=torch.jit.script(mod);args.output.parent.mkdir(parents=True,exist_ok=True);scripted.save(str(args.output))
    loaded=torch.jit.load(str(args.output));loaded.eval()
    with torch.no_grad():q=loaded(torch.from_numpy(src),torch.from_numpy(dst)).numpy()
    replay=float(np.max(np.abs(q-ref)))
    if replay>0.05:raise AssertionError(replay)
    nparams=sum(p.numel() for p in mod.parameters());index_bytes=node.nbytes+coords.astype(np.float32).nbytes+np.asarray(ck['bound_extra_mu'],dtype=np.float32).nbytes+np.asarray(ck['bound_extra_sd'],dtype=np.float32).nbytes
    print(json.dumps({'status':'completed','seed':args.seed,'validation_best_seed':best,'parity_max_abs_m':max_abs,'jit_replay_max_abs_m':replay,
                      'parameter_count':nparams,'model_parameter_bytes_fp32':4*nparams,'cached_index_bytes_including_coords_stats':int(index_bytes),
                      'jit_file_bytes':args.output.stat().st_size,'output':str(args.output)},indent=2))
if __name__=='__main__':main()

"""Export the frozen Jinan Dir-LandmarkNN comparator as node-ID -> distance TorchScript.

E4 helper only. It preserves the already-trained validation-selected checkpoint and the same
64-float directed-landmark + coordinates online state used by the frozen baseline. Online
forward includes node/coordinate gathers, normalization, cosine/L2 feature construction,
MLP forward, and distance decode. No extra normalized node table is cached.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]

def sequential_state_dict(state):
    """Adapt the saved LandmarkNN wrapper namespace to its inner Sequential namespace only.

    The frozen checkpoint was saved from LandmarkNN, whose weights are keyed as net.0.weight,
    net.2.weight, ... . This exporter reconstructs only the inner Sequential module, so strip
    exactly one leading 'net.' when present. Tensor values and model structure are unchanged.
    """
    out={}
    for k,v in state.items():
        kk=k[4:] if k.startswith('net.') else k
        if kk in out: raise RuntimeError(f'duplicate state key after namespace normalization: {kk}')
        out[kk]=v
    return out

class LandmarkNNEndToEnd(nn.Module):
    def __init__(self,node,coords,input_dim,max_distance,state):
        super().__init__()
        node=np.asarray(node,dtype=np.float32); coords=np.asarray(coords,dtype=np.float32)
        nmu=node.mean(0,dtype=np.float64).astype(np.float32); nsd=node.std(0,dtype=np.float64).astype(np.float32); nsd=np.where(nsd<1e-8,1.,nsd).astype(np.float32)
        cmu=coords.mean(0,dtype=np.float64).astype(np.float32); csd=coords.std(0,dtype=np.float64).astype(np.float32); csd=np.where(csd<1e-8,1.,csd).astype(np.float32)
        self.register_buffer('node',torch.from_numpy(node),persistent=True)
        self.register_buffer('coords',torch.from_numpy(coords),persistent=True)
        self.register_buffer('node_mu',torch.from_numpy(nmu),persistent=True)
        self.register_buffer('node_sd',torch.from_numpy(nsd),persistent=True)
        self.register_buffer('coord_mu',torch.from_numpy(cmu),persistent=True)
        self.register_buffer('coord_sd',torch.from_numpy(csd),persistent=True)
        self.max_distance=float(max_distance)
        self.net=nn.Sequential(nn.Linear(input_dim,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,1))
        self.net.load_state_dict(sequential_state_dict(state))

    def forward(self,src:torch.Tensor,dst:torch.Tensor)->torch.Tensor:
        u=src.long(); v=dst.long()
        a=(self.node.index_select(0,u)-self.node_mu)/self.node_sd
        b=(self.node.index_select(0,v)-self.node_mu)/self.node_sd
        ca=(self.coords.index_select(0,u)-self.coord_mu)/self.coord_sd
        cb=(self.coords.index_select(0,v)-self.coord_mu)/self.coord_sd
        den=torch.clamp(torch.sqrt((a*a).sum(1)*(b*b).sum(1)),min=1e-12)
        cos=((a*b).sum(1)/den).unsqueeze(1)
        l2=torch.sqrt(torch.clamp(((ca-cb)*(ca-cb)).sum(1),min=0.0)).unsqueeze(1)
        x=torch.cat((a,b,ca,cb,cos,l2),1)
        return self.net(x).squeeze(1)*self.max_distance

def numpy_features(node,coords,pairs):
    lm=node.mean(0,keepdims=True); ls=node.std(0,keepdims=True); ls=np.where(ls<1e-8,1.,ls)
    cm=coords.mean(0,keepdims=True); cs=coords.std(0,keepdims=True); cs=np.where(cs<1e-8,1.,cs)
    land=(node-lm)/ls; xy=(coords-cm)/cs; u,v=np.asarray(pairs,dtype=np.int64).T
    a,b=land[u],land[v]; ca,cb=xy[u],xy[v]
    den=np.maximum(np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1)),1e-12)
    cos=(np.sum(a*b,axis=1)/den)[:,None].astype(np.float32)
    l2=np.sqrt(np.square(ca-cb).sum(1,keepdims=True)).astype(np.float32)
    return np.concatenate((a,b,ca,cb,cos,l2),1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',type=Path,default=ROOT/'reports/audit-r4f-20260914/directed_landmarknn_alt32.json')
    ap.add_argument('--index',type=Path,default=ROOT/'results/audit-r4e-20260914/directed_landmark_residual/ALT32_index.npz')
    ap.add_argument('--data',type=Path,default=ROOT/'data/protocol_r4e/Jinan_native_directed_workload_500k.npz')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--meta-output',type=Path,required=True)
    args=ap.parse_args()
    rec=json.loads(args.report.read_text()); best=min(rec['runs'],key=lambda x:float(x['best_validation_mre_percent']))
    ckpath=Path(best['checkpoint']); ckpath=ckpath if ckpath.is_absolute() else ROOT/ckpath
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    iz=np.load(args.index); node=np.asarray(iz['node_features'],dtype=np.float32)
    z=np.load(args.data); coords=np.asarray(z['coordinates'],dtype=np.float32)
    if len(node)!=len(coords): raise RuntimeError((node.shape,coords.shape))
    if 'landmarks' in ck and 'landmarks' in iz and not np.array_equal(np.asarray(ck['landmarks']),np.asarray(iz['landmarks'])): raise RuntimeError('landmarks mismatch')
    input_dim=int(ck['input_dim']); maxd=float(ck['max_distance']); seq_state=sequential_state_dict(ck['state_dict'])
    mod=LandmarkNNEndToEnd(node,coords,input_dim,maxd,ck['state_dict']).eval()
    pairs=np.asarray(z['validation'][:4096,:2],dtype=np.int64); X=numpy_features(node,coords,pairs)
    refnet=nn.Sequential(nn.Linear(input_dim,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,1)); refnet.load_state_dict(seq_state); refnet.eval()
    with torch.no_grad():
        ref=(refnet(torch.from_numpy(X)).squeeze(1)*maxd).numpy()
        pred=mod(torch.from_numpy(pairs[:,0]),torch.from_numpy(pairs[:,1])).numpy()
    max_abs=float(np.max(np.abs(ref-pred))); max_rel=float(np.max(np.abs(ref-pred)/np.maximum(np.abs(ref),1e-9)))
    if max_abs>0.05 or max_rel>1e-4: raise AssertionError((max_abs,max_rel))
    scripted=torch.jit.script(mod); args.output.parent.mkdir(parents=True,exist_ok=True); scripted.save(str(args.output))
    loaded=torch.jit.load(str(args.output)); loaded.eval()
    with torch.no_grad(): replay=float(np.max(np.abs(loaded(torch.from_numpy(pairs[:,0]),torch.from_numpy(pairs[:,1])).numpy()-ref)))
    if replay>0.05: raise AssertionError(replay)
    nparams=sum(p.numel() for p in mod.parameters()); stats_bytes=(mod.node_mu.numel()+mod.node_sd.numel()+mod.coord_mu.numel()+mod.coord_sd.numel())*4
    meta={'status':'completed','validation_best_seed':int(best['seed']),'checkpoint':str(ckpath),'parity_max_abs':max_abs,'jit_replay_max_abs':replay,'parameter_count':int(nparams),'model_parameter_bytes_fp32':int(4*nparams),'cached_index_bytes_including_coords_and_stats':int(node.nbytes+coords.nbytes+stats_bytes),'per_node_primary_index_bytes':int(node.shape[1]*4+coords.shape[1]*4),'jit_file_bytes':int(args.output.stat().st_size),'output':str(args.output)}
    args.meta_output.parent.mkdir(parents=True,exist_ok=True); args.meta_output.write_text(json.dumps(meta,indent=2)+'\n'); print('R4N_E4_LANDMARKNN_JIT',json.dumps(meta),flush=True)
if __name__=='__main__': main()

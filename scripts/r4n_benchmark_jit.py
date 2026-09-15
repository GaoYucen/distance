"""End-to-end latency/throughput benchmark for the frozen R4M TorchScript module.

Times node-id input -> distance output, including index gathers, L/U, feature construction,
MLP, and decode. GPU timing includes H2D and D2H. Reports the last 5 of 10 runs,
matching the Survey convention, plus small-batch online latency.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch


def mean_sd(x):
    a=np.asarray(x[-5:],dtype=np.float64)
    return float(a.mean()),float(a.std(ddof=1) if len(a)>1 else 0.)

def make_batch(ids,b,seed):
    rng=np.random.default_rng(seed)
    if b<=len(ids):
        ix=rng.choice(len(ids),size=b,replace=False)
        return ids[ix]
    reps=(b+len(ids)-1)//len(ids)
    x=np.tile(ids,(reps,1))[:b].copy(); rng.shuffle(x); return x

def bench(model,ids,device,batch_sizes,runs):
    model=model.to(device); model.eval(); out=[]
    # warm up
    w=make_batch(ids,min(1024,len(ids)),17)
    s=torch.from_numpy(w[:,0].astype(np.int64)); d=torch.from_numpy(w[:,1].astype(np.int64))
    with torch.no_grad():
        if device.type=='cuda':
            _=model(s.to(device),d.to(device)).cpu(); torch.cuda.synchronize(device)
        else:_=model(s,d)
    for b in batch_sizes:
        x=make_batch(ids,b,1000+b); src_cpu=torch.from_numpy(x[:,0].astype(np.int64)); dst_cpu=torch.from_numpy(x[:,1].astype(np.int64))
        times=[]
        with torch.no_grad():
            for r in range(runs):
                if device.type=='cuda':torch.cuda.synchronize(device)
                t0=time.perf_counter()
                if device.type=='cuda':
                    src=src_cpu.to(device,non_blocking=True); dst=dst_cpu.to(device,non_blocking=True)
                    y=model(src,dst).to('cpu',non_blocking=True)
                    torch.cuda.synchronize(device)
                else:y=model(src_cpu,dst_cpu)
                # force materialization on CPU as part of end-to-end completion
                _=float(y.reshape(-1)[0])
                times.append(time.perf_counter()-t0)
        m,s=mean_sd(times); out.append({'batch_size':b,'latency_us_per_query_mean':1e6*m/b,'latency_us_per_query_sd':1e6*s/b,
                                        'throughput_mqps_mean':b/m/1e6,'run_seconds':times})
        print('R4N_LATENCY',device.type,b,out[-1],flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],required=True); ap.add_argument('--batch_sizes',default='1,32,1024,100000,1000000'); ap.add_argument('--runs',type=int,default=10); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    z=np.load(args.data); ids=z['test'][:,:2].astype(np.int64); gt=z['test'][:,2].astype(np.float64)
    dev=torch.device(args.device); model=torch.jit.load(str(args.model),map_location=dev); model.eval()
    # accuracy replay on the real test rows
    pred=[]
    with torch.no_grad():
        for i in range(0,len(ids),16384):
            s=torch.from_numpy(ids[i:i+16384,0]); d=torch.from_numpy(ids[i:i+16384,1])
            if dev.type=='cuda': p=model(s.to(dev),d.to(dev)).cpu().numpy()
            else:p=model(s,d).numpy()
            pred.append(p.reshape(-1))
    pred=np.concatenate(pred).astype(np.float64); mre=float(100*np.mean(np.abs(pred-gt)/gt))
    batches=[int(x) for x in args.batch_sizes.split(',') if x.strip()]
    rows=bench(model,ids,dev,batches,args.runs)
    rec={'status':'completed','device':args.device,'model':str(args.model),'data':str(args.data),'test_rows':int(len(ids)),'accuracy_mre_percent':mre,'runs':args.runs,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rec,indent=2)+'\n'); print('R4N_LATENCY_COMPLETE',json.dumps(rec),flush=True)

if __name__=='__main__':main()

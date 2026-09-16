"""R2 fixed-budget, scale-matched decoder pilot on paired native Jinan.
This is a mechanism pilot, not reproduction of the survey leaderboard or SOTA.
The scalar decoder multiplier matches the initial mean prediction to training
mean only; it is frozen, recorded, and never uses validation/test labels.
All five models have the same SAGE architecture and initial trainable weights.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.rgnndist2vec import RGNNdist2vec
from utils.audit_protocol import assert_disjoint_od
from scripts.audit_pilot import ordered, test_metrics
MODES=['l1','tilde_63_1','tilde_62_2','linf_symmetric','linf_asymmetric']


def decode(delta,mode):
    if mode=='l1':return delta.abs().sum(1,keepdim=True)
    if mode=='tilde_63_1':return delta[:,:63].abs().sum(1,keepdim=True)+delta[:,63:].sum(1,keepdim=True)
    if mode=='tilde_62_2':return delta[:,:62].abs().sum(1,keepdim=True)+delta[:,62:].sum(1,keepdim=True)
    if mode=='linf_symmetric':return delta.abs().amax(1,keepdim=True)
    if mode=='linf_asymmetric':return delta.amax(1,keepdim=True)
    raise ValueError(mode)


class SharedDecoder(nn.Module):
    def __init__(self,coords,edges,mode):
        super().__init__();self.mode=mode
        self.encoder=RGNNdist2vec(n_input=2,n_hidden_1=128,n_hidden_2=64,layer_type='sage',
            node_attributes=coords,edge_attributes=np.column_stack([edges,np.ones(len(edges))]),
            max_distance=1.,disable_edge_weight=True,directed=True)
        self.register_buffer('decoder_scale',torch.tensor(1.,dtype=torch.float32))
    def encode(self):
        e=self.encoder;return e.encode(e.node_features,e.edge_index,e.edge_weight)
    def forward(self,ids,z=None):
        if z is None:z=self.encode()
        return self.decoder_scale*decode(z[ids[:,1]]-z[ids[:,0]],self.mode)


def predict(model,ids):
    model.eval();out=[]
    with torch.no_grad():
        z=model.encode()
        for b in range(0,len(ids),8192):out.append(model(ids[b:b+8192],z).cpu().numpy().ravel())
    return np.concatenate(out)


def train(mode,seed,case,z,epochs,checkpoint_dir,device='cuda'):
    tr=z['train'][:20000].copy();va=z['validation'][:5000].copy();te=z['test'][:5000].copy()
    assert_disjoint_od(tr,va,te)
    assert [len(tr),len(va),len(te)]==[20000,5000,5000]
    torch.manual_seed(seed);np.random.seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    m=SharedDecoder(z['coordinates'],z['edges'],mode).to(device)
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.parameters())).hexdigest()
    scale=float(np.max(tr[:,2:4]));assert scale>0
    ti,ty=ordered(tr);vi,vy=ordered(va)
    ti=torch.as_tensor(ti,device=device);vi=torch.as_tensor(vi,device=device)
    target=torch.as_tensor(ty/scale,device=device)
    # Include both directions for the first 4096 TRAIN groups in calibration.
    ci,cy=ordered(tr[:4096]);ci=torch.as_tensor(ci,device=device)
    raw_mean=float(predict(m,ci).mean());target_mean=float(np.mean(cy/scale))
    assert raw_mean>0 and np.isfinite(raw_mean)
    factor=target_mean/raw_mean
    with torch.no_grad():m.decoder_scale.fill_(factor)
    matched_mean=float(predict(m,ci).mean())
    assert abs(matched_mean-target_mean)<1e-6
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    loss_fn=nn.SmoothL1Loss();gen=torch.Generator().manual_seed(seed)
    history=[];best=float('inf');state=None;best_epoch=0
    start=time.perf_counter()
    for ep in range(1,epochs+1):
        m.train();order=torch.randperm(len(ti),generator=gen).to(device);total=0.
        for begin in range(0,len(order),2048):
            ix=order[begin:begin+2048];opt.zero_grad(set_to_none=True)
            loss=loss_fn(m(ti[ix]),target[ix])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(ix)
        pv=predict(m,vi)*scale
        value=float(100*np.mean(np.abs(pv-vy.ravel())/vy.ravel()))
        history.append({'epoch':ep,'train_loss':total/len(ti),'validation_mre_percent':value})
        if value<best:
            best,best_epoch=value,ep
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    assert state is not None
    m.load_state_dict(state)
    # Only now construct test prediction inputs; checkpoint selection is complete.
    qi,_=ordered(te);qi=torch.as_tensor(qi,device=device)
    pt=predict(m,qi)*scale
    short=float(np.quantile((tr[:,2]+tr[:,3])/2,.25))
    result={'case':case,'mode':mode,'seed':seed,'epochs':epochs,
        'initial_trainable_weights_sha256':initial_hash,'decoder_scale':factor,
        'calibration_raw_mean':raw_mean,'calibration_target_mean':target_mean,
        'calibration_matched_mean':matched_mean,'train_distance_scale_m':scale,
        'parameter_count':sum(p.numel() for p in m.parameters()),
        'best_validation_epoch':best_epoch,'best_validation_mre_percent':best,
        'best_epoch_in_last_five':best_epoch>epochs-5,'history':history,
        'test':test_metrics(pt,te,short),'runtime_seconds':time.perf_counter()-start}
    dest=checkpoint_dir/case;dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'{mode}_s{seed}.pt'
    if path.exists():raise FileExistsError(path)
    torch.save({'state_dict':state,'mode':mode,'scale':scale,'best_epoch':best_epoch},path)
    result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(dest/f'{mode}_s{seed}_test.npz',predictions=pt,queries=te)
    # Fresh CPU instance, independent direct decoder formulas, no training.
    fresh=SharedDecoder(z['coordinates'],z['edges'],mode)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    fresh.load_state_dict(saved['state_dict']);fresh.eval()
    with torch.no_grad():
        emb=fresh.encode().numpy().astype(np.float64)
    ids,_=ordered(te);delta=emb[ids[:,1]]-emb[ids[:,0]]
    if mode=='l1':calc=np.sum(np.abs(delta),axis=1)
    elif mode.startswith('tilde_'):
        r=int(mode.split('_')[1]);calc=np.sum(np.abs(delta[:,:r]),axis=1)+np.sum(delta[:,r:],axis=1)
    elif mode=='linf_symmetric':calc=np.max(np.abs(delta),axis=1)
    else:calc=np.max(delta,axis=1)
    cpu=calc*float(fresh.decoder_scale)*scale
    np.testing.assert_allclose(cpu,pt,rtol=2e-5,atol=.1)
    replay_metrics=test_metrics(cpu,te,short)
    error=abs(replay_metrics['all']['mre_percent']-result['test']['all']['mre_percent'])
    if error>.005:raise AssertionError('CPU/GPU MRE disagreement')
    result['cpu_replay']={'passed':True,'max_prediction_difference_m':float(np.max(np.abs(cpu-pt))),
        'mre_difference_pp':error,'decoder':'independent NumPy float64 formula'}
    del m,opt,fresh,qi,ti,vi,target
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def unit_checks():
    d=torch.zeros(2,64);d[:,0]=1;d[:,63]=2
    assert decode(d,'tilde_63_1')[0].item()==3
    assert decode(-d,'tilde_63_1')[0].item()==-1
    for mode in ['l1','linf_symmetric']:
        torch.testing.assert_close(decode(d,mode),decode(-d,mode))
    assert decode(d,'linf_asymmetric')[0].item()!=decode(-d,'linf_asymmetric')[0].item()
    with np.testing.assert_raises(ValueError):
        assert_disjoint_od(np.array([[0,1,1,2]]),np.array([[1,0,2,1]]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    unit_checks();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    report_dir=ROOT/'reports/audit-r2-20260913';out=report_dir/'pilot.json'
    if out.exists():raise FileExistsError(out)
    audit=json.loads((report_dir/'data_audit.json').read_text());assert audit['passed']
    checkpoints=ROOT/'results/audit-r2-20260913/checkpoints'
    cases=['Jinan_native_directed','Jinan_native_undirected']
    report={'status':'running','planned_runs':30,'completed_runs':0,'runs':[],
        'classification':'fixed-budget mechanism pilot, not SOTA or exact reproduction of survey benchmark',
        'protocol':{'train_groups':20000,'validation_groups':5000,'test_groups':5000,
            'seeds':[42,99,1234],'epochs':args.epochs,'hidden':128,'embedding_dim':64,'batch_size':2048,
            'loss':'SmoothL1','optimizer':'Adam','learning_rate':.001,
            'calibration':'frozen multiplier learned from first 4096 training groups only, both directions',
            'selection':'validation MRE only','negative_predictions':'raw, not clipped',
            'directed_vs_undirected':'same original nodes and OD groups; target labels and graph adjacency differ'},
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_hashes':{}}
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    hashes={}
    for case in cases:
        path=ROOT/audit['cases'][case]['file']
        actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==audit['cases'][case]['sha256']
        report['data_hashes'][case]=actual;z=np.load(path)
        hashes[case]=hashlib.sha256(np.vstack([z[k][:20000 if k=='train' else 5000,:2] for k in ['train','validation','test']]).tobytes()).hexdigest()
        for seed in [42,99,1234]:
            for mode in MODES:
                r=train(mode,seed,case,z,args.epochs,checkpoints,args.device)
                report['runs'].append(r);report['completed_runs']=len(report['runs']);save()
                print('R2_RUN',case,mode,seed,'MRE',r['test']['all']['mre_percent'],'VAL_EPOCH',r['best_validation_epoch'],flush=True)
    assert len(set(hashes.values()))==1,'Paired control OD identities differ'
    for case in cases:
        for seed in [42,99,1234]:
            rows=[r for r in report['runs'] if r['case']==case and r['seed']==seed]
            assert len(set(r['initial_trainable_weights_sha256'] for r in rows))==1
            assert len(set(r['parameter_count'] for r in rows))==1
    report['summary']={}
    for case in cases:
        report['summary'][case]={}
        for mode in MODES:
            rr=[r for r in report['runs'] if r['case']==case and r['mode']==mode]
            values=np.array([r['test']['all']['mre_percent'] for r in rr])
            report['summary'][case][mode]={'mean_test_mre_percent':float(values.mean()),'std_test_mre_pp':float(values.std(ddof=1)),
                'per_seed_mre_percent':values.tolist(),'best_validation_epochs':[r['best_validation_epoch'] for r in rr],
                'negative_prediction_fraction':[r['test']['negative_prediction_fraction'] for r in rr],
                'direction_A_rmse_m_mean':float(np.mean([r['test']['all']['directional_difference_rmse'] for r in rr])),
                'high_asymmetry_mre_mean':float(np.mean([r['test']['high_asymmetry_ge_20pct']['mre_percent'] for r in rr])) if rr[0]['test']['high_asymmetry_ge_20pct']['count'] else None}
    report['status']='completed';report['same_od_control_passed']=True;report['all_30_cpu_replays_passed']=True;save()
    summary={k:report[k] for k in ['status','completed_runs','protocol','summary','same_od_control_passed','all_30_cpu_replays_passed']}
    (report_dir/'pilot_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('R2_PILOT_COMPLETE',json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()

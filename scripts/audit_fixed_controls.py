"""Two exploratory controls selected AFTER inspecting the first R1 pilot.
Not a confirmatory holdout. No hyperparameter search or overwrite of R1 runs.
Reuse the exact audited train_one loop with temporary in-process class substitution:
(1) signed max -> absolute max; (2) scalar Softplus -> scalar identity.
Production modules are not modified. Classes and weights otherwise stay identical.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_pilot as engine

OriginalShared=engine.SharedDistance
OriginalCross=engine.CrossEncoder

class SymmetricMax(OriginalShared):
    def forward(self,pairs,embeddings=None):
        z=self.encode() if embeddings is None else embeddings
        return (z[pairs[:,1]]-z[pairs[:,0]]).abs().amax(1,keepdim=True)

class LinearScalar(OriginalCross):
    def decode(self,out):
        return out if self.mode=='scalar' else super().decode(out)


def run_control(name,seed,train,val,test,coords,edges,device,checkpoint_dir):
    """Class substitution is scoped to this call and always undone."""
    try:
        if name=='shared_symmetric_linf':
            engine.SharedDistance=SymmetricMax
            mode='shared_linf'
        elif name=='cross_scalar_linear':
            engine.CrossEncoder=LinearScalar
            mode='cross_scalar'
        else:
            raise ValueError(name)
        result=engine.train_one(mode,seed,train,val,test,coords,edges,
              epochs=30,hidden=128,batch_size=2048,device=device,
              checkpoint_dir=checkpoint_dir)
        result['engine_mode']=mode
        result['mode']=name
        result['control_class']=name
        result['comparison_is_posthoc_exploratory']=True
        return result
    finally:
        engine.SharedDistance=OriginalShared
        engine.CrossEncoder=OriginalCross


def replay_control(run,q,coords,edges,directory):
    """Independently reload each new checkpoint on CPU and compare predictions."""
    mode=run['engine_mode'];seed=run['seed']
    path=directory/f'{mode}_seed{seed}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==run['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    ids,_=engine.ordered(q)
    if run['mode']=='shared_symmetric_linf':
        m=SymmetricMax(coords,edges,mode,hidden=128)
        x=torch.as_tensor(ids)
    else:
        m=LinearScalar(4,hidden=128,output_dim=64,mode='scalar')
        c=(coords-saved['coord_mean'])/saved['coord_std']
        x=torch.as_tensor(np.hstack([c[ids[:,0]],c[ids[:,1]]]).astype(np.float32))
    m.load_state_dict(saved['state_dict']);m.eval()
    with torch.no_grad():
        z=m.encode() if isinstance(m,SymmetricMax) else None
        pieces=[]
        for i in range(0,len(x),2048):
            pieces.append((m(x[i:i+2048],z) if z is not None else m(x[i:i+2048])).numpy().ravel())
    actual=np.concatenate(pieces)*saved['scale']
    prior=np.load(directory/f'{mode}_seed{seed}_test.npz')
    np.testing.assert_array_equal(prior['queries'],q)
    error=float(np.max(np.abs(actual-prior['predictions'])))
    n=len(q);du,dv=q[:,2],q[:,3]
    mre=float(50*np.mean(np.abs(actual[:n]-du)/du+np.abs(actual[n:]-dv)/dv))
    delta=abs(mre-run['test']['all']['mre_percent'])
    assert error<.1 and delta<.01,(error,delta)
    return {'max_cpu_gpu_prediction_difference_m':error,'cpu_mre_percent':mre,
            'cpu_gpu_mre_difference_pp':delta,'passed':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(2)
    data=ROOT/'data/audit/protocol_r1/splits.npz'
    assert hashlib.sha256(data.read_bytes()).hexdigest()=='212df844b130ee4f2e85845f76437ff9e647e72c45f87bd2ffd359a4dd1d8943'
    z=np.load(data);train=z['train'][:20000].copy();val=z['validation'][:5000].copy();test=z['test'][:5000].copy()
    out=ROOT/'reports/audit-r1-20260913/fixed_controls.json'
    if out.exists():raise FileExistsError(out)
    saved_dir=ROOT/'results/audit-r1-20260913/fixed_controls'
    saved_dir.mkdir(parents=True,exist_ok=False)
    report={'classification':'posthoc exploratory fixed controls, not confirmatory benchmark',
        'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'loop_source_sha256':hashlib.sha256((ROOT/'scripts/audit_pilot.py').read_bytes()).hexdigest(),
        'code_commit_at_run':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'protocol':{'train_pairs':20000,'validation_pairs':5000,'test_pairs':5000,'both_directions':True,
            'epochs':30,'hidden':128,'batch_size':2048,'learning_rate':.001,'seeds':[42,99,1234],
            'model_selection':'validation MRE only, unchanged from R1'},'runs':[],'status':'running'}
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save();start=time.perf_counter()
    # Architecture guards without touching experimental seeds (train_one resets them).
    m=SymmetricMax(z['coordinates'],z['edges'],'shared_linf',hidden=128)
    ij=torch.tensor([[0,1],[2,3],[5,6]])
    with torch.no_grad():torch.testing.assert_close(m(ij),m(ij.flip(1)))
    assert LinearScalar(4,hidden=8,mode='scalar').decode(torch.tensor([[-2.]])).item()==-2.
    del m
    for seed in [42,99,1234]:
        for name in ['shared_symmetric_linf','cross_scalar_linear']:
            # Separate control names in directories prevent aliasing original pilot checkpoints.
            dest=saved_dir/name
            result=run_control(name,seed,train,val,test,z['coordinates'],z['edges'],args.device,dest)
            result['cpu_checkpoint_replay']=replay_control(result,test,z['coordinates'],z['edges'],dest)
            report['runs'].append(result);save()
            print('FIXED_CONTROL',name,seed,result['test']['all']['mre_percent'],flush=True)
    report['summary']={}
    for name in ['shared_symmetric_linf','cross_scalar_linear']:
        rs=[r for r in report['runs'] if r['mode']==name]
        a=np.array([r['test']['all']['mre_percent'] for r in rs])
        report['summary'][name]={'mean_test_mre_percent':float(a.mean()),'std_test_mre_pp':float(a.std(ddof=1)),
            'per_seed_mre_percent':a.tolist(),'negative_prediction_fractions':[r['test']['negative_prediction_fraction'] for r in rs]}
    report['status']='completed';report['runtime_seconds']=time.perf_counter()-start;save()
    print('FIXED_CONTROL_SUMMARY',json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()

"""Exploratory small-graph pilot. Not historical Harbin/Beijing reproduction.
Every unordered OD group stays within one split, both directions are supervised,
validation MRE selects the checkpoint, test is evaluated after restoring it.
"""
from pathlib import Path
import argparse, copy, hashlib, json, subprocess, sys, time
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.audit_cross_encoder import CrossEncoder
from utils.asymmetric_metrics import L1Tilde
from utils.audit_protocol import assert_disjoint_od

MODES=['shared_l1','shared_tilde_63_1','shared_tilde_62_2','shared_linf',
       'cross_scalar','cross_l1','cross_tilde_62_2','cross_tilde_2_62']

class SharedDistance(nn.Module):
    def __init__(self,coords,edges,mode,hidden=128):
        super().__init__()
        from models.rgnndist2vec import RGNNdist2vec
        edge_attributes=np.column_stack([edges,np.ones(len(edges))])
        self.encoder=RGNNdist2vec(n_input=2,n_hidden_1=hidden,n_hidden_2=64,
            layer_type='sage',node_attributes=coords,edge_attributes=edge_attributes,
            max_distance=1.,disable_edge_weight=True,directed=True)
        self.mode=mode
        self.metric=L1Tilde(63,1) if mode=='shared_tilde_63_1' else L1Tilde(62,2)
    def encode(self):
        e=self.encoder
        return e.encode(e.node_features,e.edge_index,e.edge_weight)
    def forward(self,pairs,embeddings=None):
        z=self.encode() if embeddings is None else embeddings
        x,y=z[pairs[:,0]],z[pairs[:,1]]
        if self.mode=='shared_l1':
            return torch.abs(y-x).sum(1,keepdim=True)
        if self.mode=='shared_linf':
            return (y-x).amax(1,keepdim=True)
        return self.metric(x,y)


def ordered(q):
    ids=q[:,:2].astype(np.int64)
    return np.vstack([ids,ids[:,::-1]]).copy(),np.concatenate([q[:,2],q[:,3]]).astype(np.float32).reshape(-1,1)


def predict(model,x,batch_size=8192):
    model.eval()
    result=[]
    with torch.no_grad():
        z=model.encode() if isinstance(model,SharedDistance) else None
        for i in range(0,len(x),batch_size):
            y=model(x[i:i+batch_size],embeddings=z) if z is not None else model(x[i:i+batch_size])
            result.append(y.detach().cpu().numpy())
    return np.concatenate(result).ravel()


def test_metrics(pred,q,short_threshold):
    n=len(q);pu,pv=pred[:n],pred[n:]
    du,dv=q[:,2],q[:,3]
    per_pair=.5*(np.abs(pu-du)/du+np.abs(pv-dv)/dv)
    a=(du-dv)/2; pa=(pu-pv)/2
    alpha=np.abs(du-dv)/((du+dv)/2)
    def summary(mask):
        if not np.any(mask):return {'count':0}
        return {'count':int(mask.sum()),'mre_percent':float(100*np.mean(per_pair[mask])),
            'mae':float(np.mean(.5*(np.abs(pu[mask]-du[mask])+np.abs(pv[mask]-dv[mask])))),
            'directional_difference_rmse':float(np.sqrt(np.mean((pa[mask]-a[mask])**2)))}
    return {'all':summary(np.ones(n,dtype=bool)),
        'short_train_q25':summary((du+dv)/2<=short_threshold),
        'high_asymmetry_ge_20pct':summary(alpha>=.2),
        'negative_prediction_fraction':float(np.mean(pred<0)),
        'best_possible_symmetric_pairwise_mre_percent':float(100*np.mean(np.abs(du-dv)/(2*np.maximum(du,dv))))}


def train_one(mode,seed,train,val,test,coords,edges,epochs=30,hidden=128,
              batch_size=2048,device='cpu',checkpoint_dir=None):
    assert_disjoint_od(train,val,test)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Reduce avoidable nondeterminism; paired seed comparison is still exploratory.
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    scale=float(np.max(train[:,2:4]))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid training scale')
    mean,std=coords.mean(0),coords.std(0)
    if np.any(std==0):raise ValueError('Constant coordinate dimension')
    c=(coords-mean)/std
    shared=mode.startswith('shared_')
    if shared:
        model=SharedDistance(coords,edges,mode,hidden)
    else:
        name='scalar' if mode=='cross_scalar' else 'l1' if mode=='cross_l1' else 'l1tilde'
        r,s=(2,62) if mode=='cross_tilde_2_62' else (62,2)
        model=CrossEncoder(4,hidden,64,name,r,s)
    model.to(device)
    def features(q):
        ij,y=ordered(q)
        x=ij if shared else np.hstack([c[ij[:,0]],c[ij[:,1]]]).astype(np.float32)
        return torch.as_tensor(x,device=device),y
    xt,yt=features(train)
    xv,yv=features(val)
    yt=torch.as_tensor(yt/scale,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=nn.SmoothL1Loss()
    history=[];best=float('inf');best_epoch=0;best_state=None
    generator=torch.Generator().manual_seed(seed)
    start=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm=torch.randperm(len(xt),generator=generator).to(device)
        total_loss=0.
        for begin in range(0,len(perm),batch_size):
            ix=perm[begin:begin+batch_size]
            if len(ix)==1 and not shared:continue
            optimizer.zero_grad(set_to_none=True)
            output=model(xt[ix])
            loss=criterion(output,yt[ix])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step()
            total_loss+=float(loss.detach())*len(ix)
        pv=predict(model,xv)*scale
        vmre=float(np.mean(np.abs(pv-yv.ravel())/yv.ravel()))
        history.append({'epoch':epoch+1,'train_loss':total_loss/len(xt),'validation_mre_percent':100*vmre})
        if vmre<best:
            best,best_epoch=vmre,epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    # Test enters the prediction path only AFTER validation checkpoint selection.
    x_test,_=features(test)
    pt=predict(model,x_test)*scale
    short=float(np.quantile((train[:,2]+train[:,3])/2,.25))
    result={'mode':mode,'seed':seed,'epochs':epochs,'hidden':hidden,
        'best_validation_epoch':best_epoch,'best_validation_mre_percent':100*best,
        'test':test_metrics(pt,test,short),'history':history,
        'train_pair_count':len(train),'validation_pair_count':len(val),'test_pair_count':len(test),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'training_and_evaluation_seconds':time.perf_counter()-start,
        'negative_predictions_policy':'unclipped for signed metrics; scalar head uses explicit Softplus',
        'scale_from_train_only':scale,'device':str(device)}
    if checkpoint_dir is not None:
        dest=Path(checkpoint_dir);dest.mkdir(parents=True,exist_ok=True)
        path=dest/f'{mode}_seed{seed}.pt'
        torch.save({'state_dict':best_state,'mode':mode,'seed':seed,'hidden':hidden,
                    'scale':scale,'best_validation_epoch':best_epoch,'coord_mean':mean,'coord_std':std},path)
        np.savez_compressed(dest/f'{mode}_seed{seed}_test.npz',predictions=pt,queries=test)
        result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    del model,optimizer
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result


def main(default_family='all'):
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/audit/protocol_r1/splits.npz')
    p.add_argument('--output',default='reports/audit-r1-20260913/pilot.json')
    p.add_argument('--checkpoint-dir',default='results/audit-r1-20260913/checkpoints')
    p.add_argument('--device',default='cpu')
    p.add_argument('--family',choices=['all','shared','cross'],default=default_family)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,99,1234])
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--train-pairs',type=int,default=20000)
    p.add_argument('--eval-pairs',type=int,default=5000)
    args=p.parse_args()
    torch.set_num_threads(2)
    z=np.load(args.data)
    train=z['train'][:args.train_pairs].copy()
    val=z['validation'][:args.eval_pairs].copy()
    test=z['test'][:args.eval_pairs].copy()
    modes=[m for m in MODES if args.family=='all' or m.startswith(args.family)]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(f'Refusing to overwrite a prior pilot: {out}')
    result={'status':'running','stage':'exploratory pilot, not final benchmark',
        'dataset':'legacy_Chengdu_1901','data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'protocol':vars(args),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'results':[],'completed_runs':0,'planned_runs':len(modes)*len(args.seeds)}
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    for seed in args.seeds:
        for mode in modes:
            run=train_one(mode,seed,train,val,test,z['coordinates'],z['edges'],args.epochs,args.hidden,
                          device=args.device,checkpoint_dir=args.checkpoint_dir)
            result['results'].append(run);result['completed_runs']=len(result['results']);save()
            print('PILOT_RUN',mode,seed,'test_mre_percent',run['test']['all']['mre_percent'],flush=True)
    result['summary']={}
    for mode in modes:
        values=np.array([r['test']['all']['mre_percent'] for r in result['results'] if r['mode']==mode])
        result['summary'][mode]={'mean_test_mre_percent':float(values.mean()),
                                 'std_test_mre_pp':float(values.std(ddof=1)) if len(values)>1 else None,
                                 'seeds':len(values)}
    result['status']='completed';save();print('PILOT_COMPLETE',json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()

"""Fixed-symmetry R3: no base retraining, no test-based model selection."""
from pathlib import Path
import sys,json,hashlib,argparse,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.r3_potential_core import mre,select_shrink,node_design,fit_l2,fit_mre_lp,optimal_pair_correction
from scripts.train_survey_r2 import SharedDecoder
from scripts.audit_pilot import test_metrics
from utils.audit_protocol import assert_disjoint_od
R=ROOT/'reports/audit-r3-20260913';O=ROOT/'results/audit-r3-20260913'
SEEDS=[42,99,1234];REPS=['free','linear','mlp'];OBJS=['A_l2','MRE']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load():
 path=ROOT/'data/protocol_r2/Jinan_native_directed.npz'
 assert digest(path)=='22640a0e3fc7d5e35ae9b9d5dab6e3e172b787fd99c3b6ccac80809161b2053c'
 z=np.load(path);q={'train':z['train'][:20000],'val':z['validation'][:5000],'test':z['test'][:5000],'secondary':z['test'][5000:10000]}
 assert_disjoint_od(*q.values());return z,q
def evaluate(q,s,p,comp,cuts):
 S,A=q[:,2:4].mean(1),(q[:,2]-q[:,3])/2
 bins=np.searchsorted(cuts,S,side='left');u,v=q[:,:2].astype(int).T
 masks={'all':np.ones(len(q),bool),'identifiable':comp[u]==comp[v],'unidentifiable':comp[u]!=comp[v],'high_asym':abs(2*A)/S>=.2}
 masks.update({f'Q{i+1}':bins==i for i in range(4)});out={}
 for k,m in masks.items():
  if not m.any():out[k]={'n':0};continue
  a,b=q[m,2],q[m,3];x,y=s[m]+p[m],s[m]-p[m]
  ae=float(np.mean((p[m]-A[m])**2));en=float(np.mean(A[m]**2));se=float(np.mean((s[m]-S[m])**2))
  assert np.isclose(np.mean(((x-a)**2+(y-b)**2)/2),ae+se,rtol=1e-10,atol=1e-5)
  _,bound=optimal_pair_correction(s[m],a,b)
  out[k]={'n':int(m.sum()),'mre':100*mre(s[m],p[m],a,b),'mae':float(np.mean((abs(x-a)+abs(y-b))/2)),
   'negative_fraction':float(np.mean(np.r_[x,y]<0)),'A_mse':ae,'S_mse':se,'explained_A_energy':None if en==0 else 1-ae/en,
   'label_dependent_querywise_lower_bound_mre':float(100*np.mean(bound))}
 return out
def prepare():
 z,q=load();O.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=True)
 _,comp,keep=node_design(q['train'][:,:2],len(z['coordinates']));cuts=np.quantile(q['train'][:,2:4].mean(1),[.25,.5,.75])
 coords=z['coordinates'].astype(np.float64);coords=(coords-coords.mean(0))/coords.std(0)
 np.savez_compressed(O/'common.npz',comp=comp,keep=keep,cuts=cuts,coords=coords.astype(np.float32))
 prior=json.loads((ROOT/'reports/audit-r2-20260913/pilot.json').read_text());report={'base':[]}
 for seed in SEEDS:
  p=ROOT/f'results/audit-r2-20260913/checkpoints/Jinan_native_directed/l1_s{seed}.pt'
  old=next(r for r in prior['runs'] if r['case']=='Jinan_native_directed' and r['mode']=='l1' and r['seed']==seed)
  assert digest(p)==old['checkpoint_sha256'];ck=torch.load(p,map_location='cpu',weights_only=False)
  model=SharedDecoder(z['coordinates'],z['edges'],'l1');model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False)
  with torch.no_grad():emb=model.encode().numpy().astype(np.float64)
  mult=float(model.decoder_scale)*ck['scale'];ss={}
  for name,qs in q.items():
   u,v=qs[:,:2].astype(int).T;ss[name]=abs(emb[v]-emb[u]).sum(1)*mult
  oldpred=np.load(p.with_name(p.stem+'_test.npz'));np.testing.assert_array_equal(oldpred['queries'],q['test'])
  np.testing.assert_allclose(np.r_[ss['test'],ss['test']],oldpred['predictions'],rtol=2e-5,atol=.1)
  std=emb.std(0);std[std<1e-12]=1
  np.savez_compressed(O/f'base{seed}.npz',emb=emb,feat=(emb-emb.mean(0))/std,scale=ck['scale'],**ss)
  report['base'].append({'seed':seed,'path':str(p.relative_to(ROOT)),'sha256':digest(p),
    'metrics':{name:evaluate(qs,ss[name],np.zeros(len(qs)),comp,cuts) for name,qs in q.items()}})
 save(R/'preparation.json',report)
 save(R/'protocol.json',{'seeds':SEEDS,'train':20000,'val':5000,'test':5000,'secondary':5000,
  'secondary_not_untouched':True,'backbone_frozen':True,'dimensions':65,'capacity_not_matched':True,
  'representations':REPS,'objectives':OBJS,'MLP':'2-64-64-1, zero output layer, 300 full-batch Adam steps, lr .003, minimum TRAIN objective incl step0',
  'free_fallback':'zero for cross-training-component queries; not globally metric-preserving wrapper',
  'shrink':'one validation-only global lambda in [0,1] exact weighted-median minimization; zero included',
  'negative_predictions':'raw, no clipping','LP':'training-only HiGHS-IPM, time limit120s; report optimum only after primal/dual checks',
  'source_commit':'9577919a8973facc320eae80c8bc27194fe1d0cc','no_test_tuning':True})
 print('R3_PREPARED',flush=True)
def record(seed,rep,obj,h,info):
 z,q=load();common=np.load(O/'common.npz');comp,cuts=common['comp'],common['cuts'];base=np.load(O/f'base{seed}.npz')
 def diff(qs):
  u,v=qs[:,:2].astype(int).T;p=h[v]-h[u]
  return np.where(comp[u]==comp[v],p,0.) if rep=='free' else p
 p=diff(q['val']);lam=select_shrink(base['val'],p,q['val'][:,2],q['val'][:,3])
 assert mre(base['val'],lam*p,q['val'][:,2],q['val'][:,3])<=mre(base['val'],np.zeros(len(p)),q['val'][:,2],q['val'][:,3])+1e-10
 name=f'{rep}_{obj}_{seed}';artifact=O/f'{name}.npz';assert not artifact.exists()
 np.savez_compressed(artifact,h=h,lam=lam);again=np.load(artifact)
 out={'seed':seed,'rep':rep,'objective':obj,'solver':info,'lambda':lam,'raw':{},'shrunk':{},'artifact_sha256':digest(artifact),'replay_passed':True}
 for split,qs in q.items():
  p=diff(qs);s=base[split]
  out['raw'][split]=evaluate(qs,s,p,comp,cuts);out['shrunk'][split]=evaluate(qs,s,lam*p,comp,cuts)
  u,v=qs[:,:2].astype(int).T;pp=again['h'][v]-again['h'][u]
  if rep=='free':pp=np.where(comp[u]==comp[v],pp,0.)
  np.testing.assert_array_equal(pp,p)
  assert abs(100*mre(s,pp,qs[:,2],qs[:,3])-out['raw'][split]['all']['mre'])<1e-9
  np.testing.assert_allclose((s+p+s-p)/2,s,rtol=1e-14,atol=1e-9)
 save(R/f'{name}.json',out)
 print('R3_FIT',name,'lambda',lam,'raw',out['raw']['test']['all']['mre'],'shrunk',out['shrunk']['test']['all']['mre'],flush=True)
def convex():
 z,q=load();tr=q['train'];u,v=tr[:,:2].astype(int).T;X,c,keep=node_design(tr[:,:2],len(z['coordinates']))
 for seed in SEEDS:
  base=np.load(O/f'base{seed}.npz');scale=float(base['scale']);a,b=tr[:,2:4].T/scale;s=base['train']/scale
  for rep,design in [('free',X),('linear',base['feat'][v]-base['feat'][u])]:
   for obj in OBJS:
    theta,info=fit_l2(design,(a-b)/2) if obj=='A_l2' else fit_mre_lp(design,s,a,b,120.)
    if theta is None or not info['success']:
     save(R/f'FAILED_{rep}_{obj}_{seed}.json',info);continue
    if rep=='free':h=np.zeros(len(z['coordinates']));h[keep]=theta
    else:h=base['feat']@theta
    np.testing.assert_allclose(h[v]-h[u],np.asarray(design@theta).ravel(),rtol=1e-7,atol=1e-8)
    record(seed,rep,obj,h*scale,info)
 print('R3_CONVEX_FINISHED',flush=True)
class Potential(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x).reshape(-1)
def fit_mlp(coords,tr,s,scale,obj,seed,device,steps=300):
 """Training-only function; no held-out data argument."""
 torch.manual_seed(seed)
 if device=='cuda':torch.cuda.manual_seed_all(seed)
 model=Potential().to(device);x=torch.as_tensor(coords,dtype=torch.float32,device=device)
 u,v=[torch.as_tensor(c,dtype=torch.long,device=device) for c in tr[:,:2].astype(int).T]
 a,b=[torch.as_tensor(c/scale,dtype=torch.float32,device=device) for c in tr[:,2:4].T]
 sym=torch.as_tensor(s/scale,dtype=torch.float32,device=device)
 opt=torch.optim.Adam(model.parameters(),lr=.003);best=float('inf');state=None;history=[];beststep=0;t=time.perf_counter()
 for step in range(steps+1):
  opt.zero_grad(set_to_none=True);h=model(x);p=h[v]-h[u]
  loss=((p-(a-b)/2)**2).mean() if obj=='A_l2' else (.5*(abs(sym+p-a)/a+abs(sym-p-b)/b)).mean()
  assert torch.isfinite(loss);value=float(loss.detach());history.append(value)
  if value<best:
   best=value;beststep=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if step<steps:loss.backward();opt.step()
 model.load_state_dict(state);model.eval()
 with torch.no_grad():h=model(x).cpu().numpy().astype(np.float64)*scale
 replay=Potential();replay.load_state_dict(state);replay.eval()
 with torch.no_grad():h2=replay(torch.as_tensor(coords,dtype=torch.float32)).numpy().astype(np.float64)*scale
 h-=h.mean();h2-=h2.mean();np.testing.assert_allclose(h2,h,rtol=2e-5,atol=.03)
 return h,{'success':True,'solver':'fixed Adam MLP','steps':steps,'lr':.003,'history':history,
  'best_training_step':beststep,'training_objective':best,'seconds':time.perf_counter()-t,
  'parameters':sum(p.numel() for p in model.parameters()),'cpu_replay_error_m':float(np.max(abs(h-h2)))},state
def guards():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 coords=np.random.default_rng(4).normal(size=(12,2)).astype(np.float32);u,v=np.triu_indices(12,1)
 d=np.linalg.norm(coords[u]-coords[v],axis=1)+1;q=np.column_stack([u,v,d+.2,d]);s=d+.1
 h1,i1,w1=fit_mlp(coords,q[:40],s[:40],10.,'MRE',42,'cpu',8)
 held=q[40:].copy();held[:,2:]*=7
 h2,i2,w2=fit_mlp(coords,q[:40],s[:40],10.,'MRE',42,'cpu',8)
 assert i1['history']==i2['history'];assert all(torch.equal(w1[k],w2[k]) for k in w1)
 np.testing.assert_array_equal(h1,h2)
 save(R/'training_guard.json',{'passed':True,'training_only_signature':True,'exact_state_and_history_equal':True,
  'changed_unrelated_test_labels':True,'threads':1,'deterministic_algorithms':True})
 torch.use_deterministic_algorithms(False);torch.set_num_threads(2)
def neural():
 assert torch.cuda.is_available();z,q=load();common=np.load(O/'common.npz')
 for seed in SEEDS:
  base=np.load(O/f'base{seed}.npz')
  for obj in OBJS:
   path=O/f'mlp_{obj}_{seed}.pt';assert not path.exists()
   h,info,state=fit_mlp(common['coords'],q['train'],base['train'],float(base['scale']),obj,seed,'cuda')
   torch.save({'state_dict':state,'scale':float(base['scale'])},path);info['weights_sha256']=digest(path)
   record(seed,'mlp',obj,h,info)
 print('R3_NEURAL_FINISHED',flush=True)
def finalize(output_name='summary.json'):
 prep=json.loads((R/'preparation.json').read_text());rows=[];missing=[]
 for base in prep['base']:assert digest(ROOT/base['path'])==base['sha256']
 for seed in SEEDS:
  for rep in REPS:
   for obj in OBJS:
    p=R/f'{rep}_{obj}_{seed}.json'
    if not p.exists():missing.append(p.name);continue
    x=json.loads(p.read_text());assert digest(O/f'{rep}_{obj}_{seed}.npz')==x['artifact_sha256'] and x['replay_passed'];rows.append(x)
 summary={'status':'completed' if not missing else 'completed_with_failures','fits':len(rows),'missing':missing,
  'base_hashes_unchanged':True,'all_completed_replays_passed':True,'baseline':{},'methods':{}}
 for split in ['test','secondary']:
  vs=[b['metrics'][split]['all']['mre'] for b in prep['base']]
  summary['baseline'][split]={'mean':float(np.mean(vs)),'sd':float(np.std(vs,ddof=1)),'seeds':vs}
 for rep in REPS:
  for obj in OBJS:
   rs=[r for r in rows if r['rep']==rep and r['objective']==obj];entry={'n':len(rs),'lambdas':[r['lambda'] for r in rs]}
   for version in ['raw','shrunk']:
    entry[version]={}
    for split in ['train','val','test','secondary']:
     values=[r[version][split]['all']['mre'] for r in rs]
     entry[version][split]={'mean':float(np.mean(values)) if values else None,'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
      'seeds':values,'negative_fraction':[r[version][split]['all']['negative_fraction'] for r in rs],
      'Q1_mre_mean':float(np.mean([r[version][split]['Q1']['mre'] for r in rs])) if values else None,
      'high_asym_mre_mean':float(np.mean([r[version][split]['high_asym']['mre'] for r in rs])) if values else None}
   summary['methods'][f'{rep}/{obj}']=entry
 save(R/output_name,summary);print('R3_FINAL_SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','guards','convex','neural','finalize'],required=True)
 args=p.parse_args();torch.set_num_threads(2);globals()[args.phase]()

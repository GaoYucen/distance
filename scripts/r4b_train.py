"""R4 v1.0 stage B: full-matrix fitting, not generalization. 120 independent fits."""
import json
import subprocess
import time
import traceback
import numpy as np
import torch
from r4b_data import ROOT, CASES, save_json, sha
from r4b_models import MODES, SEEDS, UPSTREAM_SHA, SeedBatch, digest_array, metrics
STEPS=1500


def component_stats(c,mask):
    v=c[mask]
    counts=torch.bincount(v.argmax(-1),minlength=c.shape[-1]).double()/int(mask.sum())
    return {'raw_component_negative_fraction':float((v<0).double().mean()),
            'dominant_component_fractions':counts.cpu().tolist(),
            'components_above_one_percent':int((counts>.01).sum())}


def fit_group(name,mode,z,result_dir,report_dir):
    D=z['distances'];triples=z['triples'];n=len(D)
    mask_np=~np.eye(n,dtype=bool);mask=torch.as_tensor(mask_np,device='cuda')
    scale=float(D[mask_np].mean())
    target=torch.as_tensor(D/scale,dtype=torch.float32,device='cuda')
    model=SeedBatch(n,mode).cuda()
    initial=[digest_array(t) for t in model.table.detach().cpu().numpy()]
    model.calibrate(mask)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    best_loss=torch.full((3,),float('inf'),device='cuda')
    best_table=model.table.detach().clone();best_alpha=model.raw_alpha.detach().clone()
    best_step=torch.zeros(3,dtype=torch.int64,device='cuda')
    curve=torch.empty(STEPS+1,3,device='cuda');logs=[[],[],[]]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for step in range(STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        pred,c=model(return_components=True)
        loss=(pred[:,mask]-target[mask]).square().mean(-1)
        if not torch.isfinite(loss).all():raise FloatingPointError((name,mode,step))
        curve[step]=loss.detach();improved=loss.detach()<best_loss
        with torch.no_grad():
            best_loss=torch.minimum(best_loss,loss.detach())
            best_table.copy_(torch.where(improved[:,None,None],model.table.detach(),best_table))
            best_alpha.copy_(torch.where(improved,model.raw_alpha.detach(),best_alpha))
            best_step.copy_(torch.where(improved,step,best_step))
        if step%50==0:
            p=pred.detach().cpu().numpy().astype(np.float64)*scale
            for i in range(3):
                logs[i].append({'step':step,'fitting_mse_float32':float(loss[i]),
                    **metrics(p[i],D,triples),**component_stats(c[i].detach(),mask)})
            if step%250==0:print('R4B_PROGRESS',name,mode,step,loss.detach().cpu().tolist(),flush=True)
        if step==STEPS:break
        loss.sum().backward();optimizer.step()
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated();all_losses=curve.detach().cpu().numpy()
    with torch.no_grad():
        model.table.copy_(best_table);model.raw_alpha.copy_(best_alpha)
        predicted,raw=model(return_components=True)
    predicted=predicted.cpu().numpy()*scale
    tables=best_table.cpu().numpy();alphas=best_alpha.cpu().numpy();cal=model.calibration.cpu().numpy()
    bsteps=best_step.cpu().numpy();bloss=best_loss.cpu().numpy()
    np.testing.assert_array_equal(bloss,all_losses.min(0))
    np.testing.assert_array_equal(bsteps,all_losses.argmin(0))
    key=name+'__'+mode
    artifact=result_dir/(key+'.npz');metadata=report_dir/'runs'/(key+'.json')
    if artifact.exists() or metadata.exists():raise FileExistsError(key)
    np.savez_compressed(artifact,tables=tables,raw_alpha=alphas,calibration=cal,predictions=predicted,
        best_steps=bsteps,best_fitting_mse=bloss,label_mean=scale,loss_curve=all_losses)
    rows=[]
    for i,seed in enumerate(SEEDS):
        rows.append({'case':name,'mode':mode,'seed':seed,'initial_table_sha256':initial[i],
            'best_step':int(bsteps[i]),'best_normalized_fitting_mse':float(bloss[i]),
            'final_normalized_fitting_mse':float(all_losses[-1,i]),
            'best_in_last_100_updates':bool(bsteps[i]>=1400),'calibration':float(cal[i]),
            'raw_alpha':float(alphas[i]) if mode=='IQE-maxmean' else None,
            'node_scalar_count':n*64,'node_table_bytes':n*64*4,
            'decoder_trainable_bytes':4 if mode=='IQE-maxmean' else 0,'fixed_scale_metadata_bytes':12,
            'fit_metrics':metrics(predicted[i].astype(np.float64),D,triples),
            'component_stats':component_stats(raw[i],mask),'history_every_50':logs[i]})
    record={'status':'completed','case':name,'mode':mode,'runs':rows,
        'artifact':str(artifact.relative_to(ROOT)),'artifact_sha256':sha(artifact),
        'three_seed_group_wall_seconds':seconds,'group_peak_cuda_memory_bytes':peak,
        'timing_scope':'three independent seeds vectorized; includes metrics/transfers, excludes final serialization'}
    save_json(metadata,record)
    print('R4B_GROUP_COMPLETE',name,mode,'MRE',[r['fit_metrics']['mre_percent'] for r in rows],
        'SECONDS',seconds,flush=True)
    del model,optimizer,curve,best_table,best_alpha,pred,c,raw,predicted
    torch.cuda.empty_cache()
    return record


def main():
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=ROOT/'reports/audit-r4b-20260914';result=ROOT/'results/audit-r4b-20260914'
    if (report/'progress.json').exists() or result.exists():raise FileExistsError('Refusing to overwrite R4B')
    data=json.loads((report/'data_manifest.json').read_text())
    check=json.loads((report/'implementation_checks.json').read_text());assert check['passed']
    result.mkdir(parents=True);(report/'runs').mkdir()
    source_hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'scripts').glob('r4b_*.py'))}
    info={'status':'running','completed_fits':0,'planned_fits':120,'groups':[],
        'classification':'full-matrix fitting; NOT held-out performance',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hashes':source_hashes,'upstream_commit':UPSTREAM_SHA,
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'protocol':{'scalars_per_node':64,'dtype':'float32','seeds':list(SEEDS),'updates':STEPS,
            'learning_rate':.01,'batch':'all ordered non-diagonal pairs',
            'loss':'MSE of distances divided by fitting mean','selection':'minimum fitting MSE steps0..1500',
            'initialization':'CPU Normal(0,0.1), same seed table for all decoders',
            'calibration':'frozen output multiplier setting initial normalized prediction mean to1',
            'independent_seed_vectorization':True,'seed_loss_aggregation':'sum','warmstart':'not used'}}
    save_json(report/'progress.json',info);start=time.perf_counter()
    try:
        for case in CASES:
            path=ROOT/data['graphs'][case]['file'];assert sha(path)==data['graphs'][case]['sha256']
            z=np.load(path,allow_pickle=False)
            for mode in MODES:
                r=fit_group(case,mode,z,result,report);info['completed_fits']+=3
                info['groups'].append({'case':case,'mode':mode,'wall_seconds':r['three_seed_group_wall_seconds'],
                    'mre_percent':[x['fit_metrics']['mre_percent'] for x in r['runs']]})
                save_json(report/'progress.json',info)
        assert all(sha(ROOT/f)==h for f,h in source_hashes.items())
        info['status']='completed_pending_independent_replay';info['wall_seconds']=time.perf_counter()-start
        save_json(report/'progress.json',info);print('R4B_ALL_120_FITS_COMPLETE',info['wall_seconds'],flush=True)
    except Exception as error:
        info['status']='failed';info['failure_type']=type(error).__name__;info['traceback']=traceback.format_exc()
        save_json(report/'progress.json',info)
        raise

if __name__=='__main__':main()

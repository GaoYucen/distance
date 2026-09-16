"""R4D: certified structural compression of the frozen R4C bidirectional ALT32 dictionary.

No neural training and no validation/test-driven grouping. See frozen protocol in
`docs/r4d-certified-landmark-compression-20260914/PROTOCOL.md`.
"""
from __future__ import annotations
import hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.r4c_realroads as r4c

REPORT = ROOT / 'reports/audit-r4d-20260914'
RESULT = ROOT / 'results/audit-r4d-20260914'
R_VALUES = (1, 2, 3, 4, 5, 6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def morton_codes(coords: np.ndarray) -> np.ndarray:
    """Deterministic 2D Morton codes after per-axis min-max quantization to uint16."""
    c = np.asarray(coords, dtype=np.float64)
    lo, hi = c.min(0), c.max(0)
    span = np.maximum(hi-lo, 1e-12)
    q = np.rint((c-lo)/span*65535).astype(np.uint32)
    def spread(x):
        x = x & np.uint32(0x0000ffff)
        x = (x | (x << np.uint32(8))) & np.uint32(0x00FF00FF)
        x = (x | (x << np.uint32(4))) & np.uint32(0x0F0F0F0F)
        x = (x | (x << np.uint32(2))) & np.uint32(0x33333333)
        x = (x | (x << np.uint32(1))) & np.uint32(0x55555555)
        return x.astype(np.uint64)
    return spread(q[:,0]) | (spread(q[:,1]) << np.uint64(1))


def hypercube_pairs(r: int):
    signs = np.array(np.meshgrid(*[[-1.,1.]]*r, indexing='ij')).reshape(r,-1).T
    # Representatives have first coordinate +1; every opposite pair appears once.
    reps = signs[signs[:,0] > 0]
    reps = reps[np.lexsort(tuple(reps[:,j] for j in range(r-1,-1,-1)))]
    assert len(reps) == 2**(r-1)
    return reps


def block_from_teachers(Fpair: np.ndarray, r: int):
    """Walsh first-order LS projection. Fpair rows alternate f_l, g_l per landmark."""
    landmarks_per_block = 2**(r-1)
    if Fpair.shape[0] != 2*landmarks_per_block:
        raise ValueError((Fpair.shape, r))
    reps = hypercube_pairs(r)
    rows=[]; funcs=[]
    for i,s in enumerate(reps):
        rows.extend((s, -s))
        funcs.extend((Fpair[2*i], Fpair[2*i+1]))
    signs=np.asarray(rows,dtype=np.float64)
    F=np.asarray(funcs,dtype=np.float64)
    F_center=F-F.mean(axis=1,keepdims=True)
    A=np.concatenate((np.ones((len(signs),1)), signs),axis=1)
    gram=A.T@A
    np.testing.assert_allclose(gram, len(signs)*np.eye(r+1),atol=1e-12,rtol=0)
    X=(A.T@F_center)/len(signs)  # [h,z1,...,zr] x nodes
    recon=A@X
    rmse=float(np.sqrt(np.mean((recon-F_center)**2)))
    return X,rmse


def edge_q(X: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h=X[0];z=X[1:]
    return np.abs(z[:,v]-z[:,u]).sum(0)+(h[v]-h[u])


def certify_float32(X: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray):
    raw=edge_q(X,u,v)
    c0=max(1.0,float(np.max(raw/w)))
    # Quantize to the stated deployment representation and leave a conservative safety margin.
    X32=(X/(c0*(1+3e-6))).astype(np.float32)
    q=edge_q(X32.astype(np.float64),u,v)
    c1=max(1.0,float(np.max(q/w)))
    if c1>1:
        X32=(X32/(c1*(1+3e-6))).astype(np.float32)
        q=edge_q(X32.astype(np.float64),u,v)
    ratio=float(np.max(q/w))
    margin=float(np.min(w-q))
    tol=1e-5*np.maximum(1.,w)
    if np.any(q>w+tol):
        raise AssertionError(('edge certificate failed',ratio,float(np.max(q-w-tol))))
    return X32, {'pre_quantization_scale':c0,'post_quantization_scale':c1,
                 'max_edge_ratio':ratio,'minimum_edge_slack_m':margin}


def decode_blocks(blocks, ids):
    u,v=np.asarray(ids,dtype=np.int64).T
    pred=np.zeros(len(ids),dtype=np.float64)
    winners=np.full(len(ids),-1,dtype=np.int64)
    for k,X32 in enumerate(blocks):
        q=edge_q(X32.astype(np.float64),u,v)
        take=q>pred
        pred[take]=q[take];winners[take]=k
    return pred,winners


def ordered(q):
    ids=q[:,:2].astype(np.int64)
    return np.vstack((ids,ids[:,::-1]))


def case_edges(case, z):
    A=r4c.build_native_csr(case,z,r4c.CASES[case]['raw_edges']).tocoo()
    return A.row.astype(np.int64),A.col.astype(np.int64),A.data.astype(np.float64)


def process_case(case: str):
    info=r4c.CASES[case]
    if digest(info['path']) != info['sha256']:
        raise AssertionError('protocol R2 hash mismatch')
    z=np.load(info['path'])
    alt_path=ROOT/'results/audit-r4c-20260914'/case/'ALT32/index.npz'
    if not alt_path.exists():
        raise FileNotFoundError(alt_path)
    alt=np.load(alt_path)
    features=alt['features'].astype(np.float64)
    landmarks=alt['landmarks'].astype(np.int64)
    test=alt['test_queries'].astype(np.float64)
    np.testing.assert_array_equal(test,z['test'])
    if features.shape!=(len(z['coordinates']),64) or len(landmarks)!=32:
        raise AssertionError((features.shape,len(landmarks)))
    # Teacher potentials: f_l(v)=d(l,v); g_l(v)=-d(v,l).
    forward=features[:,:32].T
    reverse=-features[:,32:].T
    teacher_lb=alt['test_lb'].astype(np.float64)
    ids=ordered(test)
    short=float(np.quantile((z['train'][:,2]+z['train'][:,3])/2,.25))
    base_metrics=r4c.group_metrics(teacher_lb,test,short)

    # Deterministic spatial order over frozen landmarks, no validation/test labels.
    lm_coords=z['coordinates'][landmarks].astype(np.float64)
    codes=morton_codes(lm_coords)
    order=np.lexsort((landmarks,codes))
    ordered_landmarks=landmarks[order]
    ordered_forward=forward[order]
    ordered_reverse=reverse[order]
    # interleave forward and reverse per landmark
    paired=np.empty((64,features.shape[0]),dtype=np.float64)
    paired[0::2]=ordered_forward;paired[1::2]=ordered_reverse
    eu,ev,ew=case_edges(case,z)

    rows=[]
    case_dir=RESULT/case;case_dir.mkdir(parents=True,exist_ok=True)
    for r in R_VALUES:
        lp=2**(r-1);groups=32//lp
        if groups*lp!=32:raise AssertionError('nondivisible landmark count')
        blocks=[];block_meta=[]
        before_rmse=[]
        for g in range(groups):
            F=paired[2*g*lp:2*(g+1)*lp]
            X,rmse=block_from_teachers(F,r)
            X32,cert=certify_float32(X,eu,ev,ew)
            blocks.append(X32);before_rmse.append(rmse)
            block_meta.append({'group':g,'r':r,'landmarks':ordered_landmarks[g*lp:(g+1)*lp].tolist(),
                               'teacher_centered_rmse_m':rmse,**cert})
        pred,winners=decode_blocks(blocks,ids)
        # Certified lower-bound sanity on every test direction.
        truth=np.concatenate((test[:,2],test[:,3])).astype(np.float64)
        tolerance=1e-3+1e-5*np.maximum(1.,truth)
        excess=pred-truth
        if np.any(excess>tolerance):
            raise AssertionError((case,r,float(excess.max())))
        metrics=r4c.group_metrics(pred,test,short)
        diff=float(np.max(np.abs(pred-teacher_lb))) if r==1 else None
        mre_diff=metrics['mre_percent']-base_metrics['mre_percent']
        active=[float(np.mean(winners==k)) for k in range(groups)]
        scalars=groups*(r+1)
        packed=np.stack(blocks,axis=0) if len({b.shape for b in blocks})==1 else np.array(blocks,dtype=object)
        # all groups share same r and shape, so stack is expected
        artifact=case_dir/f'r{r}_B{scalars}.npz'
        np.savez_compressed(artifact,blocks=packed,ordered_landmarks=ordered_landmarks,
                            predictions=pred.astype(np.float32),test_queries=test)
        row={'r':r,'blocks':groups,'landmarks_per_block':lp,'scalars_per_node':scalars,
             'bytes_per_node_float32':4*scalars,'metrics':metrics,'mre_increase_vs_ALT32_LB_pp':float(mre_diff),
             'max_prediction_difference_vs_ALT32_LB_m':diff,
             'mean_teacher_centered_rmse_m':float(np.mean(before_rmse)),
             'max_teacher_centered_rmse_m':float(np.max(before_rmse)),
             'max_edge_ratio':float(max(m['max_edge_ratio'] for m in block_meta)),
             'max_pre_quantization_scale':float(max(m['pre_quantization_scale'] for m in block_meta)),
             'active_block_fractions':active,'block_metadata':block_meta,
             'artifact':str(artifact.relative_to(ROOT)),'artifact_sha256':digest(artifact)}
        if r==1:
            if diff is None or diff>.05 or abs(mre_diff)>.002:
                raise AssertionError(('r1 must reproduce ALT32-LB',case,diff,mre_diff))
        rows.append(row)
        print('R4D_RESULT',case,'r',r,'B',scalars,'MRE',metrics['mre_percent'],'DELTA',mre_diff,
              'SCALE',row['max_pre_quantization_scale'],flush=True)
    out={'case':case,'classification':'development compression diagnostic; test labels already inspected in R4C',
         'teacher':'R4C ALT32-LB','teacher_mre_percent':base_metrics['mre_percent'],
         'teacher_landmarks':landmarks.tolist(),'morton_ordered_landmarks':ordered_landmarks.tolist(),
         'source_alt_artifact':str(alt_path.relative_to(ROOT)),'source_alt_sha256':digest(alt_path),
         'rows':rows}
    (REPORT/f'{case}.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return out


def render(cases):
    lines=['# R4D｜有证书的双向 Landmark 势函数字典压缩','',
      '本轮没有神经训练。输入是 R4C 冻结的 ALT32 双向有向 landmark 下界；通过固定 Morton 分组、Walsh 一阶投影和全有向边缩放认证，压缩 64 个势函数。所有结果仍属于开发数据诊断。','',
      '| 每节点 scalar | 济南 MRE (%) | 深圳 MRE (%) | 济南较ALT增加(pp) | 深圳较ALT增加(pp) |','|---:|---:|---:|---:|---:|']
    by={x['case']:x for x in cases}
    jin=by['Jinan_native_directed'];sz=by['Shenzhen_native_directed_uniform']
    # baseline
    lines.append(f"| 64 (ALT32-LB teacher) | {jin['teacher_mre_percent']:.4f} | {sz['teacher_mre_percent']:.4f} | 0 | 0 |")
    for jr,sr in zip(jin['rows'],sz['rows']):
        assert jr['scalars_per_node']==sr['scalars_per_node']
        B=jr['scalars_per_node']
        lines.append(f"| {B} | {jr['metrics']['mre_percent']:.4f} | {sr['metrics']['mre_percent']:.4f} | {jr['mre_increase_vs_ALT32_LB_pp']:.4f} | {sr['mre_increase_vs_ALT32_LB_pp']:.4f} |")
    lines += ['','## 解释边界','',
      '- r=1 / B=64 必须逐查询复现 ALT32-LB；这是把每个 landmark 的 forward/reverse 两个势函数写成一个 1D L1+势差块的代数等价检查，不是性能提升。',
      '- B<64 的版本全部在 float32 量化后重新扫描完整原始有向边；边约束通过后才报告，因此对同一 SCC 的任意最短路保持不高估。',
      '- MRE 上升衡量压缩代价；它不是最终独立测试，也没有与新训练方法比较。',
      '- 如果 32 或更低 scalar 仍保留较低误差，下一步才值得用剩余预算加入可认证的残差方向/对称结构；如果压缩迅速恶化，应把贡献转向更好的字典分组与结构学习，而不是声称简单 Walsh 压缩已经足够。']
    (REPORT/'R4D_STATUS_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    REPORT.mkdir(parents=True,exist_ok=False);RESULT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cases=[]
    for case in ('Jinan_native_directed','Shenzhen_native_directed_uniform'):
        cases.append(process_case(case))
    render(cases)
    manifest={'status':'completed','classification':'certified development compression diagnostic',
              'cases':[x['case'] for x in cases],'r_values':list(R_VALUES),'runtime_seconds':time.perf_counter()-start,
              'protocol':'docs/r4d-certified-landmark-compression-20260914/PROTOCOL.md',
              'script_sha256':digest(Path(__file__))}
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('R4D_COMPLETE',manifest['runtime_seconds'],flush=True)

if __name__=='__main__':main()

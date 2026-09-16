"""CPU and independent NumPy replay of all saved R4B fits."""
import csv
import json
import time
import numpy as np
import torch
from r4b_data import ROOT, CASES, save_json, sha
from r4b_models import MODES, SEEDS, SeedBatch, numpy_decode, metrics, triangle_check


def main():
    torch.set_num_threads(2)
    report=ROOT/'reports/audit-r4b-20260914'
    progress=json.loads((report/'progress.json').read_text())
    assert progress['status']=='completed_pending_independent_replay' and progress['completed_fits']==120
    if (report/'summary.json').exists():raise FileExistsError('Summary already exists')
    data=json.loads((report/'data_manifest.json').read_text())
    checks=json.loads((report/'implementation_checks.json').read_text());assert checks['passed']
    assert all(sha(ROOT/f)==h for f,h in progress['source_hashes'].items())
    rows=[];replays=[];timings=[];initial_hashes={};start=time.perf_counter()
    for case in CASES:
        datapath=ROOT/data['graphs'][case]['file'];assert sha(datapath)==data['graphs'][case]['sha256']
        z=np.load(datapath,allow_pickle=False);D=z['distances'];n=len(D)
        for mode in MODES:
            group=json.loads((report/'runs'/(case+'__'+mode+'.json')).read_text())
            path=ROOT/group['artifact'];assert sha(path)==group['artifact_sha256']
            a=np.load(path,allow_pickle=False)
            assert a['tables'].shape==(3,n,64) and a['tables'].dtype==np.float32
            assert a['loss_curve'].shape==(1501,3)
            np.testing.assert_array_equal(a['best_fitting_mse'],a['loss_curve'].min(0))
            np.testing.assert_array_equal(a['best_steps'],a['loss_curve'].argmin(0))
            timings.append({'case':case,'mode':mode,'three_seed_wall_seconds':group['three_seed_group_wall_seconds'],
                'three_seed_peak_cuda_bytes':group['group_peak_cuda_memory_bytes']})
            for i,seed in enumerate(SEEDS):
                row=group['runs'][i];assert row['seed']==seed
                initial_hashes.setdefault((case,seed),set()).add(row['initial_table_sha256'])
                cpu=SeedBatch(n,mode,(seed,))
                with torch.no_grad():
                    cpu.table.copy_(torch.from_numpy(a['tables'][i:i+1]))
                    cpu.raw_alpha.copy_(torch.from_numpy(a['raw_alpha'][i:i+1]))
                    cpu.calibration.copy_(torch.from_numpy(a['calibration'][i:i+1]))
                    out=cpu()[0].numpy()*float(a['label_mean'])
                independent=numpy_decode(a['tables'][i],mode,float(a['raw_alpha'][i]),float(a['calibration'][i]))*float(a['label_mean'])
                saved=a['predictions'][i];tolerance=3e-5*float(a['label_mean'])
                np.testing.assert_allclose(out,saved,rtol=3e-5,atol=tolerance)
                np.testing.assert_allclose(independent,saved,rtol=3e-5,atol=tolerance)
                fm=metrics(independent,D,z['triples'])
                mre_difference=abs(fm['mre_percent']-row['fit_metrics']['mre_percent'])
                assert mre_difference<.001
                triangle=triangle_check(independent,1e-8*max(1.,float(D.max())))
                assert triangle['passed'] and independent.min()>=-1e-9
                np.testing.assert_allclose(np.diag(independent),0.,atol=1e-9)
                if case=='cycle97' and mode=='T1':assert fm['max_relative_percent']/100>=1-2/97-1e-5
                if mode=='L1':np.testing.assert_allclose(independent,independent.T,atol=1e-10)
                replays.append({'case':case,'mode':mode,'seed':seed,'passed':True,
                    'max_cpu_gpu_prediction_difference':float(np.max(np.abs(out-saved))),
                    'max_numpy_gpu_prediction_difference':float(np.max(np.abs(independent-saved))),
                    'mre_difference_pp':mre_difference,'triangle':triangle,'artifact_sha256':group['artifact_sha256']})
                rows.append({k:v for k,v in row.items() if k!='history_every_50'})
    assert len(rows)==120 and all(len(x)==1 for x in initial_hashes.values())
    summary={'status':'completed_and_replayed','fits':120,'replays':120,'source_commit':progress['source_commit'],
        'classification':progress['classification'],'protocol':progress['protocol'],
        'all_initial_seed_tables_matched_across_methods':True,'all_replays_passed':True,
        'implementation_checks':checks['checks'],'training_wall_seconds':progress['wall_seconds'],
        'replay_wall_seconds':time.perf_counter()-start,'summary':{},'timings':timings,
        'best_in_last_100_updates':sum(r['best_in_last_100_updates'] for r in rows)}
    names=['mre_percent','normalized_mse','p95_relative_percent','max_relative_percent',
        'direction_half_difference_rmse','cycle_half_difference_rmse','prediction_cycle_max_abs']
    for case in CASES:
        summary['summary'][case]={}
        for mode in MODES:
            rr=[r for r in rows if r['case']==case and r['mode']==mode]
            item={k:{'mean':float(np.mean([r['fit_metrics'][k] for r in rr])),
                'sd':float(np.std([r['fit_metrics'][k] for r in rr],ddof=1)),
                'per_seed':[r['fit_metrics'][k] for r in rr]} for k in names}
            item['best_steps']=[r['best_step'] for r in rr]
            item['active_components']=[r['component_stats']['components_above_one_percent'] for r in rr]
            summary['summary'][case][mode]=item
    save_json(report/'replay.json',{'passed':True,'replays':replays})
    save_json(report/'fit_rows.json',{'rows':rows});save_json(report/'summary.json',summary)
    with (report/'fit_metrics.csv').open('w',newline='') as handle:
        columns=['case','mode','seed','best_step','node_table_bytes','decoder_trainable_bytes']+list(rows[0]['fit_metrics'])
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for r in rows:writer.writerow({**{k:r[k] for k in columns if k in r},**r['fit_metrics']})
    lines=['# R4B｜固定64维的表示机制实验','',
        '**120/120拟合完成，120/120独立CPU与NumPy重放通过。**',
        '这些是四个合成图的全矩阵拟合。所有非对角标签均参与训练，不是泛化分数，不是新真实路网成绩。',
        f"源提交：`{progress['source_commit']}`；作者参照：`{progress['upstream_commit']}`。",'',
        '## 协议','',
        '每节点64个float32；种子42/99/1234；Adam 0.01；1500次全批量更新；按归一化拟合MSE选模型，不按MRE选。每模型固定校准初始输出均值。三种子独立参数沿额外轴向量化，已核对串行Adam。',
        'IQE使用作者区间公式与maxmean初始系数；MRN-L2是非平方L2加最大势差的解码器级节点表，不是原强化学习流程复现。没有热启动进入主比较。','',
        '## MRE（%，均值 ± 三种子样本标准差）','',
        '| 模型 | 双向树127 | 单向环97 | 有向网格144 | 积图192 |','|---|---:|---:|---:|---:|']
    for mode in MODES:
        vals=[summary['summary'][c][mode]['mre_percent'] for c in CASES]
        lines.append('| '+mode+' | '+' | '.join(f"{v['mean']:.4f} ± {v['sd']:.4f}" for v in vals)+' |')
    lines+=['','## 归一化MSE（三种子均值）','',
        '| 模型 | 双向树127 | 单向环97 | 有向网格144 | 积图192 |','|---|---:|---:|---:|---:|']
    for mode in MODES:lines.append('| '+mode+' | '+' | '.join(f"{summary['summary'][c][mode]['normalized_mse']['mean']:.7f}" for c in CASES)+' |')
    lines+=['','## 解释边界','',
        'H1比较多分量与T1；H2比较纯势差P64；H3比较Shared-L1及MRN/IQE。不能从H1成立推出H3，也不把观察结果后选出的最佳K当成预注册单方法。',
        '积图的B4、B8、Shared-L1有64坐标内精确构造，已独立核验。随机训练未达零误差不证明表示能力不足。T1在单向97环最坏方向误差下界1−2/97不是平均MRE下界。',
        f"{summary['best_in_last_100_updates']}/120配置的最小目标出现在最后100步；这不是充分收敛判据。完整1501步目标和每50步指标保留。IQE仅使用预定8维分量，没有4/16维调参。",'',
        '## 成本和证据','',
        '每节点256字节；IQE-maxmean另有4字节全局学习系数，各模型尺度元数据另计。归档含曲线和多种子，不等于部署大小。',
        f"训练墙钟共{progress['wall_seconds']:.2f}秒。各图/模型的三种子组时间和显存见summary.json，不能当作单模型独立实测时间。本轮没有生产查询延迟测量。",
        '`data_manifest.json`记录图/标签/映射/哈希；`implementation_checks.json`记录作者公式和结构检查；`runs/*.json`含逐种子每50步曲线；`fit_metrics.csv`120行；`replay.json`含逐产物哈希、重放差异与全部三点检查。',
        '服务器`results/audit-r4b-20260914/*.npz`保留检查点、预测和1501步目标。原main/VLDB和R1–R3未覆盖，PR不自动合并。']
    (report/'R4B_STATUS_ZH.md').write_text('\n'.join(lines)+'\n')
    print('R4B_ALL_REPLAYS_PASSED',len(replays),flush=True)
    print('R4B_COMPACT_MRE',json.dumps({c:{m:summary['summary'][c][m]['mre_percent'] for m in MODES} for c in CASES}),flush=True)

if __name__=='__main__':main()

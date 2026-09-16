"""Aggregate the frozen R4N E4 efficiency/storage evidence and apply a predeclared decision rule.

The direct comparator is Dir-LandmarkNN, the strongest fair baseline on all E3 confirmation graphs.
GREEN requires complete CPU/GPU main-protocol measurements, <=3x worst main-batch latency ratio,
and <=1.10x logical online-state ratio. YELLOW allows a constant-factor 3-10x latency tradeoff or
<=1.25x state ratio; RED is an order-of-magnitude latency regression, >1.25x state, missing main
measurements, or a protocol-integrity failure. These thresholds are fixed before reading E4 results.
"""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import numpy as np

MAIN_BATCHES=(100000,1000000)
GREEN_LAT=3.0
RED_LAT=10.0
GREEN_STORAGE=1.10
RED_STORAGE=1.25

def load(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return json.loads(p.read_text())

def rowmap(rec): return {int(x['batch_size']):x for x in rec['results']}

def mean_train(report):
    vals=[float(x['train_seconds']) for x in report.get('runs',[]) if x.get('train_seconds') is not None]
    return float(statistics.mean(vals)) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r4m-cpu',required=True); ap.add_argument('--r4m-gpu',required=True)
    ap.add_argument('--lnn-cpu',required=True); ap.add_argument('--lnn-gpu',required=True)
    ap.add_argument('--lnn-meta',required=True); ap.add_argument('--r4m-report',required=True); ap.add_argument('--lnn-report',required=True)
    ap.add_argument('--jinan-index',required=True); ap.add_argument('--jinan-data',required=True); ap.add_argument('--fla-index-report',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--md',required=True)
    a=ap.parse_args()
    rc,rg,lc,lg=map(load,(a.r4m_cpu,a.r4m_gpu,a.lnn_cpu,a.lnn_gpu)); lm=load(a.lnn_meta); rr=load(a.r4m_report); lr=load(a.lnn_report); fr=load(a.fla_index_report)
    for x in (rc,rg,lc,lg):
        if x.get('status')!='completed' or int(x.get('runs',0))!=10: raise RuntimeError(('bad benchmark',x.get('status'),x.get('runs')))
    maps={'cpu':(rowmap(rc),rowmap(lc)),'cuda':(rowmap(rg),rowmap(lg))}
    ratios={}; complete=True
    for dev,(rm,bm) in maps.items():
        for b in MAIN_BATCHES:
            if b not in rm or b not in bm: complete=False; continue
            ratios[f'{dev}_{b}']=float(rm[b]['latency_us_per_query_mean']/bm[b]['latency_us_per_query_mean'])
    if not complete: maxlat=float('inf')
    else: maxlat=max(ratios.values())
    iz=np.load(a.jinan_index); node=np.asarray(iz['node_features']); z=np.load(a.jinan_data); coords=np.asarray(z['coordinates']); n=len(coords)
    r4m_params=int(rr['runs'][0]['parameter_count']); lnn_params=int(lm['parameter_count'])
    r4m_stats=(64*2+2*2+5*2)*4; lnn_stats=(64*2+2*2)*4
    primary=int(node.nbytes+coords.nbytes)
    r4m_state=primary+r4m_stats+4*r4m_params; lnn_state=primary+lnn_stats+4*lnn_params
    storage_ratio=float(r4m_state/lnn_state)
    fla_n=int(fr['node_count']); fla_primary=fla_n*264; fla_r4m=fla_primary+r4m_stats+4*r4m_params; fla_lnn=fla_primary+lnn_stats+4*lnn_params
    if complete and maxlat<=GREEN_LAT and storage_ratio<=GREEN_STORAGE:
        decision='GREEN'; closed=True; claim='strengthened'; user=False
        manuscript='Close E4 with an end-to-end CPU/GPU efficiency and storage table. R4M retains its E3 accuracy advantage while staying within the predeclared constant-factor latency and near-equal online-state envelope versus Dir-LandmarkNN.'
        next_action='Proceed only to R4N-V3-03 theory/manuscript-gap audit; do not run further empirical method search.'
    elif complete and maxlat<RED_LAT and storage_ratio<=RED_STORAGE:
        decision='YELLOW'; closed=True; claim='qualified'; user=True
        manuscript='Close E4 as a qualified accuracy/efficiency tradeoff: R4M remains non-dominated in accuracy but pays a measurable constant-factor systems cost that must be stated explicitly.'
        next_action='Stop new empirical experiments and enter R4N-V3-03 manuscript audit with explicit efficiency qualification.'
    else:
        decision='RED'; closed=False; claim='weakened'; user=True
        manuscript='E4 does not support an acceptable efficiency/storage Pareto under the frozen protocol; do not hide the systems cost or change caching/timing definitions.'
        next_action='Stop automatic experiments and return to paper-level judgment; no cache/timing/model changes without user authorization.'
    rec={'status':'completed','round_id':'R4N-V3-02','evidence_slot':'E4-efficiency-pareto','predeclared_rule':{'GREEN':f'complete CPU/GPU 100K+1M and worst R4M/Dir-LandmarkNN latency ratio <= {GREEN_LAT} and logical online state ratio <= {GREEN_STORAGE}','YELLOW':f'complete, worst latency ratio < {RED_LAT} and state ratio <= {RED_STORAGE}, but GREEN thresholds missed','RED':f'missing main protocol, latency ratio >= {RED_LAT}, state ratio > {RED_STORAGE}, or protocol-integrity failure'},'benchmarks':{'R4M':{'cpu':rc,'gpu':rg},'Dir-LandmarkNN':{'cpu':lc,'gpu':lg}},'main_batch_latency_ratio_r4m_over_lnn':ratios,'worst_main_batch_latency_ratio':maxlat,'storage':{'per_node_primary_bytes_including_coordinates':264,'Jinan_nodes':n,'Jinan_R4M_logical_online_bytes':r4m_state,'Jinan_LandmarkNN_logical_online_bytes':lnn_state,'Jinan_R4M_over_LandmarkNN_ratio':storage_ratio,'R4M_model_parameter_bytes_fp32':4*r4m_params,'LandmarkNN_model_parameter_bytes_fp32':4*lnn_params,'FLA_nodes':fla_n,'FLA_primary_index_plus_coords_bytes':fla_primary,'FLA_R4M_logical_online_bytes':fla_r4m,'FLA_LandmarkNN_logical_online_bytes':fla_lnn},'offline':{'Jinan_directed_landmark_build_seconds_shared':lr.get('index_build_seconds'),'FLA_directed_landmark_build_seconds':fr.get('build_seconds'),'R4M_mean_training_seconds_per_seed':mean_train(rr),'LandmarkNN_mean_training_seconds_per_seed':mean_train(lr),'training_protocol':'both fixed at 300 seconds/seed; actual wall time reported'},'decision_state':decision,'paper_delta':{'claim_delta':claim,'evidence_slot_closed':closed,'evidence_slot_closed_name':'E4-efficiency-pareto' if closed else None,'manuscript_delta':manuscript,'remaining_paper_gaps':['E1/E5 final formal-theory and manuscript claim/limitation audit'],'decision_state':decision,'next_action':next_action,'user_judgment_required':user}}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2)+'\n')
    lines=['# R4N-V3-02 E4 efficiency/storage Pareto','',f'**Decision: {decision}**', '',f'- Worst main-batch R4M/Dir-LandmarkNN latency ratio: `{maxlat:.4f}x`',f'- Jinan logical online-state ratio: `{storage_ratio:.4f}x`',f'- FLA R4M logical online state: `{fla_r4m/1024**2:.2f} MiB`',f'- FLA directed-landmark preprocessing: `{float(fr.get("build_seconds",0)):.2f} s`','', '## Main protocol latency', '', '| Device | Batch | R4M us/query | Dir-LandmarkNN us/query | Ratio |','|---|---:|---:|---:|---:|']
    for dev,(rm,bm) in maps.items():
        for b in MAIN_BATCHES:
            lines.append(f"| {dev} | {b} | {rm[b]['latency_us_per_query_mean']:.6f} | {bm[b]['latency_us_per_query_mean']:.6f} | {rm[b]['latency_us_per_query_mean']/bm[b]['latency_us_per_query_mean']:.4f}x |")
    lines += ['', 'All timings are node-ID -> distance end-to-end. R4M includes gather, outward rounding, L/U reduction, feature construction, MLP and decode; GPU includes H2D and D2H. No extra normalized/lo/hi node tables are cached.']
    Path(a.md).write_text('\n'.join(lines)+'\n')
    print('R4N_E4_AGGREGATE',json.dumps({'decision':decision,'worst_latency_ratio':maxlat,'storage_ratio':storage_ratio,'closed':closed}),flush=True)
if __name__=='__main__': main()

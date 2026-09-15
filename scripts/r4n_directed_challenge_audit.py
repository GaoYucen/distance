"""Quantify how much of the Jinan difficulty is caused by directed/asymmetric labels.

This is a diagnostic, not a new model. It compares the Survey undirected labels and
native-directed labels on the exact same retained OD rows, and computes a hard oracle
lower bound for any symmetric predictor when both directions of a pair are evaluated.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.r4c_realroads as r4c
from scripts.r4e_build_directed_jinan_workload import label_pairs

DATA=ROOT/'data/protocol_r4e/Jinan_native_directed_workload_500k.npz'
BASE=ROOT/'data/protocol_r2/Jinan_native_directed.npz'
RAW=ROOT/'data/figshare_native_20260913/edge_jinan.csv'
SURVEY=Path('/workspace/shortest-distance-survey-r4e/data/W_Jinan/real_workload_perturb_500k/W_Jinan_test.queries.npz')
OUT=ROOT/'reports/audit-r4n-20260915/directed_challenge_jinan.json'


def main():
    z=np.load(DATA); test=z['test'].astype(np.float64); ids=test[:,:2].astype(np.int64); d=test[:,2]
    base=np.load(BASE); original=base['original_node_ids'].astype(np.int64); n=len(original)
    A=r4c.build_native_csr('Jinan_native_directed',base,RAW)
    rev=label_pairs(A,ids[:,::-1])
    alpha=np.abs(d-rev)/((d+rev)/2.0)

    # If a predictor is symmetric, it must use one scalar p for both u->v and v->u.
    # For pairwise average MRE, the exact optimum is p=min(d_uv,d_vu), yielding
    # |d_uv-d_vu|/(2*max(d_uv,d_vu)).
    sym_oracle_pair=np.abs(d-rev)/(2*np.maximum(d,rev))
    mid=(d+rev)/2.0
    sym_mid_pair=.5*(np.abs(mid-d)/d+np.abs(mid-rev)/rev)

    # Reconstruct the exact retained rows from the original Survey test split and compare
    # undirected labels to native-directed labels on the same OD sequence.
    q=np.load(SURVEY,allow_pickle=False); src=q['src'].astype(np.int64)-1; dst=q['dst'].astype(np.int64)-1
    mapping=np.full(int(original.max())+1,-1,dtype=np.int64); mapping[original]=np.arange(n)
    inmap=(src<len(mapping))&(dst<len(mapping))
    mapped=np.full((len(src),2),-1,dtype=np.int64)
    mapped[inmap,0]=mapping[src[inmap]]; mapped[inmap,1]=mapping[dst[inmap]]
    valid=inmap&(mapped[:,0]>=0)&(mapped[:,1]>=0)&(mapped[:,0]!=mapped[:,1])
    kept=mapped[valid]
    if kept.shape!=ids.shape or not np.array_equal(kept,ids):
        raise AssertionError('retained Survey OD sequence does not match frozen directed workload')
    und=q['dist'].astype(np.float64)[valid]
    label_shift=np.abs(und-d)/d

    rec={
      'status':'completed',
      'classification':'diagnostic quantification of directed-task difficulty; no model tuning',
      'test_rows':int(len(d)),
      'asymmetry':{
        'mean_alpha_percent':float(100*alpha.mean()),
        'median_alpha_percent':float(100*np.median(alpha)),
        'alpha_ge_1pct_fraction':float(np.mean(alpha>=.01)),
        'alpha_ge_10pct_fraction':float(np.mean(alpha>=.10)),
        'alpha_ge_20pct_fraction':float(np.mean(alpha>=.20)),
        'p95_alpha_percent':float(100*np.quantile(alpha,.95)),
      },
      'symmetric_predictor_paired_oracle':{
        'definition':'best possible per-unordered-pair symmetric scalar under average MRE of both directions',
        'mre_percent':float(100*sym_oracle_pair.mean()),
        'p95_pair_mre_percent':float(100*np.quantile(sym_oracle_pair,.95)),
        'midpoint_mre_percent':float(100*sym_mid_pair.mean()),
        'note':'This is a paired-direction diagnostic lower bound, not the Survey one-direction workload score.'
      },
      'same_od_undirected_to_directed_label_shift':{
        'mean_relative_change_vs_directed_percent':float(100*label_shift.mean()),
        'median_relative_change_percent':float(100*np.median(label_shift)),
        'p95_relative_change_percent':float(100*np.quantile(label_shift,.95)),
        'fraction_over_5pct':float(np.mean(label_shift>=.05)),
        'fraction_over_20pct':float(np.mean(label_shift>=.20)),
        'directed_longer_than_undirected_fraction':float(np.mean(d>und+1e-9)),
        'mean_directed_m':float(d.mean()),
        'mean_survey_undirected_m':float(und.mean()),
      }
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2)+'\n')
    print('R4N_DIRECTED_CHALLENGE',json.dumps(rec),flush=True)

if __name__=='__main__':main()

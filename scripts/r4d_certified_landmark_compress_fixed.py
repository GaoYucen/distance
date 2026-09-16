"""Execution-only fix for R4D float32 edge certification.

Scientific grouping, Walsh projection, budgets, landmarks and evaluation remain frozen.
The first run correctly stopped because a single post-quantization rescale could still
round back above an edge weight. This wrapper iterates quantize->certify->rescale
until the stored float32 block itself is a true edge lower bound.
"""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.r4d_certified_landmark_compress as base


def certify_float32_strict(X,u,v,w):
    raw=base.edge_q(X,u,v)
    c0=max(1.0,float(np.max(raw/w)))
    total_scale=c0
    X32=(X/total_scale).astype(np.float32)
    iterations=0
    ratios=[]
    while True:
        q=base.edge_q(X32.astype(np.float64),u,v)
        ratio=float(np.max(q/w));ratios.append(ratio)
        if ratio <= 1.0:
            break
        # Scale the already quantized representation, then requantize. A small fixed
        # multiplicative margin only protects against another upward float32 rounding.
        factor=ratio*(1.0+2e-6)
        total_scale*=factor
        X32=(X32/factor).astype(np.float32)
        iterations+=1
        if iterations>12:
            raise AssertionError(('float32 edge certificate did not converge',ratios))
    q=base.edge_q(X32.astype(np.float64),u,v)
    margin=float(np.min(w-q))
    if np.any(q>w):
        raise AssertionError(('strict edge certificate failed',float(np.max(q/w)),float(np.max(q-w))))
    return X32,{
        'pre_quantization_scale':c0,
        'post_quantization_scale':total_scale/c0,
        'total_scale':total_scale,
        'certification_iterations':iterations,
        'certification_ratio_trace':ratios,
        'max_edge_ratio':float(np.max(q/w)),
        'minimum_edge_slack_m':margin,
    }


def main():
    base.certify_float32=certify_float32_strict
    base.main()


if __name__=='__main__':main()

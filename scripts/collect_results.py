#!/usr/bin/env python3
"""
Collect and compare all experiment results: L1 (original) vs L1Tilde (ablation).

Reads experiment_results.json from --log_dir directories,
compares L1→L1Tilde delta, and prints a formatted table.

Usage:
    python3 collect_results.py [--base /path/to/results]
"""
import json, os, sys, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)  # l1tilde-metric-study/
DEFAULT_RESULTS = os.path.join(STUDY_DIR, "results")

parser = argparse.ArgumentParser()
parser.add_argument("--base", default=os.environ.get("RESULTS_DIR", DEFAULT_RESULTS))
args = parser.parse_args()

BASE = args.base
CITIES = ["OSM_Harbin_Small", "OSM_Harbin", "OSM_Chengdu", "OSM_Qingdao", "OSM_Beijing"]

# ---- L1 reference results (from full benchmark) ----
L1_REF = {
    "RGAT":   {"dir": "rgnndist2vec_gat"},
    "RSAGE":  {"dir": "rgnndist2vec_sage"},
    "RGCN":   {"dir": "rgnndist2vec_gcn"},
    "RNE":    {"dir": "rne"},
    "LpNorm": {"dir": "lpnorm_manhattan"},
}

# ---- L1Tilde results ----
L1T_REF = {
    "RGAT-L1Tilde":   {"dir": "rgnndist2vec_l1tilde"},
    "RSAGE-L1Tilde":  {"dir": "rgnndist2vec_l1tilde"},
    "RGCN-L1Tilde":   {"dir": "rgnndist2vec_l1tilde"},
    "RNE-L1Tilde":    {"dir": "rne_l1tilde"},
    "LpNorm-L1Tilde": {"dir": "lpnorm_l1tilde"},
}


def read_mre(dir_name):
    path = os.path.join(BASE, dir_name, "experiment_results.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return d["evaluation"]["test"]["mre_percent"]
    return None


print("=" * 90)
print("  L1 vs L1Tilde  —  Controlled Metric Ablation")
print("=" * 90)

for l1_name, l1_info in L1_REF.items():
    l1t_name = f"{l1_name}-L1Tilde"
    l1t_info = L1T_REF[l1t_name]

    print(f"\n{'─' * 80}")
    print(f"  {l1_name}  →  {l1t_name}")
    print(f"{'─' * 80}")
    print(f"  {'City':<20s}  {'L1 MRE':>10s}  {'L1Tilde MRE':>14s}  {'Delta':>10s}  {'Verdict':>20s}")
    print(f"  {'─' * 20}  {'─' * 10}  {'─' * 14}  {'─' * 10}  {'─' * 20}")

    for city in CITIES:
        l1_mre = read_mre(f"{city}_{l1_info['dir']}")
        l1t_mre = read_mre(f"{city}_{l1t_info['dir']}")

        if l1_mre and l1t_mre:
            delta = l1t_mre - l1_mre
            if delta < -1.0:
                verdict = "L1Tilde wins (significant)"
            elif delta < -0.1:
                verdict = "L1Tilde slightly better"
            elif delta > 1.0:
                verdict = "L1 wins (significant)"
            elif delta > 0.1:
                verdict = "L1 slightly better"
            else:
                verdict = "~same (within noise)"
            print(f"  {city:<20s}  {l1_mre:>8.2f}%  {l1t_mre:>12.2f}%  {delta:>+8.2f}%  {verdict:>20s}")
        elif l1_mre:
            print(f"  {city:<20s}  {l1_mre:>8.2f}%  {'N/A':>14s}  {'N/A':>10s}")
        elif l1t_mre:
            print(f"  {city:<20s}  {'N/A':>10s}  {l1t_mre:>12.2f}%  {'N/A':>10s}")
        else:
            print(f"  {city:<20s}  {'N/A':>10s}  {'N/A':>14s}  {'N/A':>10s}")

print(f"\n{'=' * 90}")
print("  Note: Negative delta = L1Tilde improves over L1")
print("=" * 90)

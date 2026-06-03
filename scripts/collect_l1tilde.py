import json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)  # l1tilde-metric-study/
base = os.environ.get("RESULTS_DIR", os.path.join(STUDY_DIR, "results"))

cities = ["OSM_Harbin_Small", "OSM_Harbin", "OSM_Chengdu", "OSM_Qingdao", "OSM_Beijing"]

# Known L1 results (from existing experiments)
l1_results = {
    "rgnndist2vec_gat":  {"OSM_Harbin_Small": 9.06, "OSM_Harbin": 5.88, "OSM_Chengdu": 3.28, "OSM_Qingdao": 2.96, "OSM_Beijing": 3.52},
    "rgnndist2vec_sage": {"OSM_Harbin_Small": 9.63, "OSM_Harbin": 6.51, "OSM_Chengdu": 5.47, "OSM_Qingdao": 3.65, "OSM_Beijing": 5.35},
    "rgnndist2vec_gcn":  {"OSM_Harbin_Small": 20.43, "OSM_Harbin": 32.02, "OSM_Chengdu": 24.81, "OSM_Qingdao": 26.48, "OSM_Beijing": 23.70},
    "rne":                {"OSM_Harbin_Small": 11.80, "OSM_Harbin": 26.11, "OSM_Chengdu": 12.74, "OSM_Qingdao": 13.76, "OSM_Beijing": 16.70},
    "lpnorm_manhattan":   {"OSM_Harbin_Small": 13.39, "OSM_Harbin": 12.21, "OSM_Chengdu": 12.81, "OSM_Qingdao": 11.50, "OSM_Beijing": 11.79},
}

print("=" * 100)
print("L1 vs L1_tilde Ablation Results")
print("=" * 100)

for model_base, gnn in [("rgnndist2vec_l1tilde", "gat"), ("rgnndist2vec_l1tilde", "sage"), ("rgnndist2vec_l1tilde", "gcn")]:
    l1tilde_name = f"{model_base}_{gnn}" if model_base == "rgnndist2vec_l1tilde" else model_base
    display_name = f"RGAT-L1Tilde" if gnn == "gat" else (f"RSAGE-L1Tilde" if gnn == "sage" else f"RGCN-L1Tilde")
    l1_name = f"rgnndist2vec_{gnn}"
    l1_display = f"RGAT-L1" if gnn == "gat" else (f"RSAGE-L1" if gnn == "sage" else f"RGCN-L1")
    
    print(f"\n{'─'*80}")
    print(f"  {l1_display}  →  {display_name}")
    print(f"{'─'*80}")
    print(f"  {'City':<20s}  {'L1 MRE':>10s}  {'L1Tilde MRE':>14s}  {'Delta':>10s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*14}  {'─'*10}")
    
    for city in cities:
        l1_mre = l1_results[l1_name].get(city, None)
        
        dir_name = f"{city}_{model_base}"
        if gnn:
            dir_name = f"{dir_name}"  # same dir, gnn_layer in name
        path = os.path.join(base, dir_name, "experiment_results.json")
        
        l1tilde_mre = None
        if os.path.exists(path):
            d = json.load(open(path))
            t = d["evaluation"]["test"]
            l1tilde_mre = t["mre_percent"]
        
        if l1_mre and l1tilde_mre:
            delta = l1tilde_mre - l1_mre
            better = "← L1Tilde better" if delta < 0 else ("← L1 better" if delta > 0 else "")
            print(f"  {city:<20s}  {l1_mre:>8.2f}%  {l1tilde_mre:>12.2f}%  {delta:>+8.2f}%  {better}")
        elif l1_mre:
            print(f"  {city:<20s}  {l1_mre:>8.2f}%  {'N/A':>14s}")
        else:
            print(f"  {city:<20s}  {'N/A':>10s}  {l1tilde_mre:>12.2f}%")

# RNE
print(f"\n{'─'*80}")
print(f"  RNE-L1  →  RNE-L1Tilde")
print(f"{'─'*80}")
print(f"  {'City':<20s}  {'L1 MRE':>10s}  {'L1Tilde MRE':>14s}  {'Delta':>10s}")
print(f"  {'─'*20}  {'─'*10}  {'─'*14}  {'─'*10}")
for city in cities:
    l1_mre = l1_results["rne"].get(city)
    path = os.path.join(base, f"{city}_rne_l1tilde", "experiment_results.json")
    l1tilde_mre = None
    if os.path.exists(path):
        d = json.load(open(path))
        t = d["evaluation"]["test"]
        l1tilde_mre = t["mre_percent"]
    if l1_mre and l1tilde_mre:
        delta = l1tilde_mre - l1_mre
        better = "← L1Tilde better" if delta < 0 else ("← L1 better" if delta > 0 else "")
        print(f"  {city:<20s}  {l1_mre:>8.2f}%  {l1tilde_mre:>12.2f}%  {delta:>+8.2f}%  {better}")

# LpNorm
print(f"\n{'─'*80}")
print(f"  LpNorm-L1  →  LpNorm-L1Tilde")
print(f"{'─'*80}")
print(f"  {'City':<20s}  {'L1 MRE':>10s}  {'L1Tilde MRE':>14s}  {'Delta':>10s}")
print(f"  {'─'*20}  {'─'*10}  {'─'*14}  {'─'*10}")
for city in cities:
    l1_mre = l1_results["lpnorm_manhattan"].get(city)
    path = os.path.join(base, f"{city}_lpnorm_l1tilde", "experiment_results.json")
    l1tilde_mre = None
    if os.path.exists(path):
        d = json.load(open(path))
        t = d["evaluation"]["test"]
        l1tilde_mre = t["mre_percent"]
    if l1_mre and l1tilde_mre:
        delta = l1tilde_mre - l1_mre
        better = "← L1Tilde better" if delta < 0 else ("← L1 better" if delta > 0 else "")
        print(f"  {city:<20s}  {l1_mre:>8.2f}%  {l1tilde_mre:>12.2f}%  {delta:>+8.2f}%  {better}")

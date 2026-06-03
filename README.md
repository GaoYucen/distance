# VLDB Distance — L1Tilde Metric Ablation Study

Code and experiments for: *"Does L̃₁ asymmetric metric improve over L1 for directed road network distance estimation?"*

## Data Preparation

### Data Sources

Road network data is downloaded from OpenStreetMap via OSMnx for 4 Chinese cities:

| City | Approx. Nodes | Approx. Edges | Query Pairs (45/node) |
|------|:-----------:|:-----------:|:-------------------:|
| Harbin | 44K | 108K | ~1,981,000 |
| Chengdu | 111K | 275K | ~4,995,000 |
| Qingdao | 119K | 294K | ~5,355,000 |
| Beijing | 163K | 402K | ~7,335,000 |

All road networks are **directed graphs** (oneway roads preserved, ~18% edges are oneway). Shortest-path distances are asymmetric: `d(u→v) ≠ d(v→u)` for ~94% of query pairs.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Download + Convert                              │
│  python scripts/prepare_data.py                           │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────┐  │
│  │ OSMnx    │ →  │ .nodes   │    │ data/OSM_{City}/   │  │
│  │ download │    │ .edges   │    │   {City}.nodes      │  │
│  │          │    │          │    │   {City}.edges      │  │
│  └──────────┘    └──────────┘    └────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Generate Query Pairs (same script)              │
│  ┌──────────────────┐    ┌─────────────────────────────┐ │
│  │ CCH shortest     │ →  │ {City}_train.queries (80%)  │ │
│  │ path (inertial   │    │ {City}_test.queries  (20%)  │ │
│  │ flow ordering)   │    │ Format: u,v,d_uv,d_vu       │ │
│  └──────────────────┘    └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Stage 3: Extra files — auto-generated on first model run │
│  RNE → .parts (METIS)    CatBoost → landmark embeddings   │
└─────────────────────────────────────────────────────────┘
```

### Query Strategy

Queries are generated proportional to graph size: **45 pairs per node** (matching the original survey protocol). For each (source, target) pair, both `d_uv` and `d_vu` are computed via CCH exact shortest path, enabling evaluation on directed graphs.

### Auto-Generated Files

When running RNE, CatBoost, or CatBoostNN for the first time on a city, the required extra data files are **automatically generated** and cached to disk:

| Model | Auto-generated file | Time (Harbin) | Dependency |
|-------|--------------------|:---:|------------|
| RNE | `.parts` (hierarchical METIS partitions) | ~1 min | `pymetis` |
| CatBoost / CatBoostNN | `landmark_dim61.embeddings` | ~3 min | (none extra) |

Subsequent runs skip generation. You can also pre-generate manually:
```bash
python scripts/generate_landmark_distances.py --data_dir data/OSM_Harbin
python scripts/generate_parts_file_rne.py --data_dir data/OSM_Harbin      # needs pymetis
```

## Quick Start

```bash
# 1. Install dependencies
pip install torch torch_geometric numpy pandas networkx scikit-learn tqdm matplotlib seaborn osmnx routingkit_cch catboost

# 2. Download and prepare data (Stages 1 + 2: .nodes, .edges, .queries)
python scripts/prepare_data.py                      # all 4 cities
# python scripts/prepare_data.py --city Harbin      # single city

# (Optional) Install pymetis if you plan to use RNE
pip install pymetis

# 4. Run a single model
python train.py --model_class rgnndist2vec --gnn_layer gat \
    --data_dir data/OSM_Harbin --query_dir data/OSM_Harbin/proportional \
    --epochs 20 --device cpu --force_shift 0 --log_dir results/test

# 5. Run full benchmark (16 baselines × 5 cities)
bash scripts/run_full_benchmark.sh

# 6. Run L1→L̃₁ ablation (5 groups × 5 cities)
bash scripts/run_l1tilde_ablation.sh

# 7. Collect & compare results
python scripts/collect_results.py
```

## Directory Structure

```
l1tilde-metric-study/
├── train.py                # Main entry point (supports all models via --model_class)
├── models/                 # All model implementations (runnable)
│   ├── basemodel.py        #   Base model class (fit / evaluate)
│   ├── geodnn.py           #   GeoDNN
│   ├── landmark.py         #   Landmark (non-ML)
│   ├── lpnorm.py           #   LpNorm (non-ML, L1 Manhattan)
│   ├── lpnorm_l1tilde.py   #   ★ LpNorm → L̃₁ variant
│   ├── rgnndist2vec.py     #   ★ RGNNdist2vec (GAT/SAGE/GCN, L1)
│   ├── rgnndist2vec_l1tilde.py  # ★ RGNNdist2vec → L̃₁ variant
│   ├── rne.py              #   ★ RNE (L1-mean)
│   ├── rne_l1tilde.py      #   ★ RNE → L̃₁ variant
│   ├── ndist2vec.py        #   NDist2Vec
│   ├── vdist2vec.py        #   VDist2Vec
│   ├── distancenn.py       #   DistanceNN
│   ├── embeddingnn.py      #   EmbeddingNN
│   ├── aneda.py            #   ANEDA
│   ├── path2vec.py         #   Path2Vec
│   ├── catboostmodel.py    #   CatBoost (GBDT)
│   ├── catboostnn.py       #   CatBoostNN
│   ├── dist2gnn_model.py   #   Dist2GNN (our method)
│   ├── dist2vec.py         #   Dist2Vec pretraining
│   └── sparse_matrix_model.py
├── utils/                  # Shared utilities
│   ├── data_utils.py       #   Graph I/O, landmark selection, preprocessing
│   ├── torch_utils.py      #   Dataset classes, optimizer, device detection
│   ├── plot_utils.py       #   Learning curves, error plots
│   ├── asymmetric_metrics.py  # L̃₁/L̃∞ definition
│   └── active_finetune.py  #   Active fine-tuning
├── scripts/                # Experiment orchestration
│   ├── prepare_data.py         # Data download (OSMnx → .nodes/.edges → CCH queries)
│   ├── run_full_benchmark.sh   # Full: 16 baselines × 5 cities = 85 exps
│   ├── run_l1tilde_ablation.sh # Ablation: 5 L1Tilde models × 5 cities
│   ├── run_l1tilde_full.sh     # Ablation (alternative version)
│   ├── collect_results.py      # Collect & compare L1 vs L1Tilde
│   └── collect_l1tilde.py      # L1Tilde result collector
├── original/               # Historical reference (flat copies, for diff comparison)
├── modified/               # Historical reference (L1Tilde variants + .diff files)
├── data/                   # Downloaded data (gitignored)
└── results/                # Experiment output (gitignored)
```

## All Supported Models

```bash
# Non-ML baselines
python train.py --model_class landmark --landmark_selection random ...
python train.py --model_class lpnorm --p_norm 1 ...

# NN baselines
python train.py --model_class geodnn ...
python train.py --model_class ndist2vec ...
python train.py --model_class vdist2vec ...
python train.py --model_class distancenn ...
python train.py --model_class embeddingnn ...
python train.py --model_class catboostnn ...
python train.py --model_class catboost ...

# Functional baselines
python train.py --model_class path2vec ...
python train.py --model_class aneda ...
python train.py --model_class rne ...

# GNN baselines (L1 metric)
python train.py --model_class rgnndist2vec --gnn_layer gat ...
python train.py --model_class rgnndist2vec --gnn_layer sage ...
python train.py --model_class rgnndist2vec --gnn_layer gcn ...

# L1→L̃₁ variants (ablation study)
python train.py --model_class rgnndist2vec_l1tilde --gnn_layer gat ...
python train.py --model_class rne_l1tilde ...
python train.py --model_class lpnorm_l1tilde ...

# Our method
python train.py --model_class dist2gnn ...
```

## Experiment Design

**Question**: What is the real contribution of L̃₁ over L1, controlling for model architecture?

**Method**: Take 5 baselines that use L1 distance (RGAT, RSAGE, RGCN, RNE, LpNorm), create L̃₁ variants with *identical architecture*, run the same 5-city directed road network protocol, measure ΔMRE.

**Theoretical basis**: Theorem 3-5 from the paper prove L1 cannot isometrically embed directed graphs, but L̃₁ can.

## Key Finding

| Model | Small cities | Large cities | Overall |
|-------|:---:|:---:|:---:|
| RGAT→L1Tilde | ~same | ~same | No significant gain |
| RSAGE→L1Tilde | ~same | **-1.4 to -1.9%** | Improves on large graphs |
| RGCN→L1Tilde | ~same | ~same | No change |
| RNE→L1Tilde | -1% | **+12 to +16%** | Degrades |
| LpNorm→L1Tilde | broken | broken | N/A (raw coords) |

**Conclusion**: L̃₁ helps SAGE (which lacks attention) on large directed graphs, but GAT's attention mechanism already captures directionality. The metric's benefit depends on the architecture's ability to utilize asymmetric dimensions.

## L̃₁ Metric Implementation (~10 lines per baseline)

The change is minimal and localized:

1. `__init__`: add `l1tilde_r`, `l1tilde_s` parameters
2. `forward()`: replace `torch.norm(emb1-emb2, p=1)` with:
   ```python
   r, s = self.l1tilde_r, self.l1tilde_s
   sym = torch.abs(emb2[:,:r] - emb1[:,:r]).sum(dim=1, keepdim=True)
   asym = (emb2[:,r:r+s] - emb1[:,r:r+s]).sum(dim=1, keepdim=True)
   distance = sym + asym
   ```

See `modified/*.diff` for exact line-by-line changes.

#!/bin/bash
set -e
# ============================================================================
# VLDB Distance — L1Tilde Ablation Script
# Replaces L1 with L̃₁ in 5 baselines (RGAT/RSAGE/RGCN/RNE/LpNorm)
# Same architecture, same protocol — isolates metric contribution.
#
# Usage:
#   bash run_l1tilde_ablation.sh
#
# Requires: modified/ code already placed in $SRC/models/
# ============================================================================

# ---- Configuration (auto-detected from script location) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # l1tilde-metric-study/

SRC="${STUDY_DIR}"                           # train.py + models/ + utils/
PYTHON="${PYTHON:-python3}"
DATA_BASE="${STUDY_DIR}/data"
LOG_BASE="${STUDY_DIR}/results"
DEVICE="${DEVICE:-}"                          # empty = train.py auto-detect
FORCE_SHIFT="${FORCE_SHIFT:-0}"

# ---- Protocol Parameters (same as full benchmark) ----
BATCH_SIZE=1024
EMB_DIM=64
EPOCHS=20
TIME_LIMIT=5
GNN_LR=0.01
GNN_LOSS=smoothl1
RNE_LR=0.003
RNE_LOSS=mse

# L1Tilde: r symmetric dims + s asymmetric dims (r+s = embed_dim)
R=62
S=2

# ---- Cities & Query Dirs ----
CITIES_DATA=(
    "OSM_Harbin_Small|random_500k"
    "OSM_Harbin|proportional"
    "OSM_Chengdu|proportional"
    "OSM_Qingdao|proportional"
    "OSM_Beijing|proportional"
)

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

COMPLETED=0

# ============================================================================
run_l1tilde() {
    local model_class=$1 city=$2 query=$3 extra_args=$4 tag=$5

    local log_dir="${LOG_BASE}/${city}_${tag}"
    local data_dir="${DATA_BASE}/${city}"
    local query_dir="${data_dir}/${query}"

    if [ -f "${log_dir}/experiment_results.json" ]; then
        echo -e "${YELLOW}[SKIP] ${tag} on ${city} — already done${NC}"
        return
    fi

    COMPLETED=$((COMPLETED + 1))
    echo -e "${GREEN}[${COMPLETED}] ${tag} on ${city} $(date '+%H:%M:%S')${NC}"

    local model_lr=$GNN_LR
    local model_loss=$GNN_LOSS
    local epochs_flag="--epochs ${EPOCHS}"
    local time_flag="--time_limit ${TIME_LIMIT}"

    case "$model_class" in
        rne_l1tilde)
            model_lr=$RNE_LR
            model_loss=$RNE_LOSS
            ;;
        lpnorm_l1tilde)
            epochs_flag="--epochs 1"
            time_flag=""
            extra_args="$extra_args --l1tilde_r 1 --l1tilde_s 1"  # 2D coords
            ;;
    esac

    local device_flag=""
    [ -n "$DEVICE" ] && device_flag="--device $DEVICE"

    $PYTHON -u train.py \
        --model_class "$model_class" \
        --data_dir "$data_dir" \
        --query_dir "$query_dir" \
        --learning_rate $model_lr \
        --loss $model_loss \
        --batch_size_train $BATCH_SIZE \
        --embedding_dim $EMB_DIM \
        $epochs_flag $time_flag --validate \
        $device_flag --force_shift $FORCE_SHIFT \
        --l1tilde_r $R --l1tilde_s $S \
        --log_dir "$log_dir" \
        $extra_args
}

# ============================================================================
cd "$SRC"

echo "============================================"
echo "  L1Tilde Ablation Experiment"
echo "  Started: $(date)"
echo "  r=$R, s=$S  (embed_dim=$EMB_DIM)"
echo "  5 L1 baselines × 5 cities = 25 experiments"
echo "============================================"

for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"

    # 1. RGAT-L1Tilde (best L1 baseline)
    run_l1tilde "rgnndist2vec_l1tilde" "$city" "$query" "--gnn_layer gat"  "rgnndist2vec_l1tilde_gat"

    # 2. RSAGE-L1Tilde
    run_l1tilde "rgnndist2vec_l1tilde" "$city" "$query" "--gnn_layer sage" "rgnndist2vec_l1tilde_sage"

    # 3. RGCN-L1Tilde
    run_l1tilde "rgnndist2vec_l1tilde" "$city" "$query" "--gnn_layer gcn"  "rgnndist2vec_l1tilde_gcn"

    # 4. RNE-L1Tilde
    run_l1tilde "rne_l1tilde" "$city" "$query" "" "rne_l1tilde"

    # 5. LpNorm-L1Tilde (non-ML, instant)
    run_l1tilde "lpnorm_l1tilde" "$city" "$query" "--p_norm 1" "lpnorm_l1tilde"
done

echo -e "\n${GREEN}===== ALL DONE =====${NC}"
echo "Run: python3 scripts/collect_results.py  to compare L1 vs L1Tilde"

#!/bin/bash
set -e
# ============================================================================
# VLDB Distance — Full Benchmark Script
# 16 baselines + Dist2GNN × 5 cities = 85 experiments
# Aligns with survey paper training protocol.
#
# Usage:
#   bash run_full_benchmark.sh
#
# Configure paths below before running.
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

# ---- Protocol Parameters (fixed, per survey paper) ----
BATCH_SIZE=1024
EMB_DIM=64
EPOCHS=20
TIME_LIMIT=5
DEFAULT_LR=0.01
PATH2VEC_LR=0.03
ANEDA_LR=0.03
RNE_LR=0.003
CATBOOSTNN_LR=0.0003
LOSS=mse
GNN_LOSS=smoothl1

# ---- Cities & Query Dirs ----
# Hrb_Small uses random_500k; large cities use proportional (45 pairs/node)
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

TOTAL_START=$(date +%s)
COMPLETED=0

# ============================================================================
run_exp() {
    local model=$1 city=$2 query=$3 extra_args=$4 tag=$5

    local model_tag="${model}${tag:+"_$tag"}"
    local log_dir="${LOG_BASE}/${city}_${model_tag}"
    local data_dir="${DATA_BASE}/${city}"
    local query_dir="${data_dir}/${query}"

    if [ -f "${log_dir}/experiment_results.json" ]; then
        echo -e "${YELLOW}[SKIP] ${model_tag} on ${city} — already done${NC}"
        return
    fi

    COMPLETED=$((COMPLETED + 1))
    echo -e "${GREEN}[${COMPLETED}] ${model_tag} on ${city} $(date '+%H:%M:%S')${NC}"

    # Model-specific LR and loss
    local model_lr=$DEFAULT_LR
    local model_loss=$LOSS
    case "$model" in
        path2vec) model_lr=$PATH2VEC_LR ;;
        aneda)    model_lr=$ANEDA_LR ;;
        rne)      model_lr=$RNE_LR ;;
        catboostnn) model_lr=$CATBOOSTNN_LR ;;
        rgnndist2vec) model_loss=$GNN_LOSS ;;
    esac

    # Non-ML models: 1 epoch, no validate
    local validate_flag="--validate"
    local epochs_flag="--epochs ${EPOCHS}"
    local time_flag="--time_limit ${TIME_LIMIT}"
    if [ "$model" == "landmark" ] || [ "$model" == "lpnorm" ]; then
        validate_flag=""
        epochs_flag="--epochs 1"
        time_flag=""
    fi

    local device_flag=""
    [ -n "$DEVICE" ] && device_flag="--device ${DEVICE}"
    [ "$model" == "catboost" ] && device_flag="--device cpu"

    $PYTHON -u train.py \
        --model_class "$model" \
        --data_dir "$data_dir" \
        --query_dir "$query_dir" \
        --learning_rate $model_lr \
        --loss $model_loss \
        --batch_size_train $BATCH_SIZE \
        --embedding_dim $EMB_DIM \
        $epochs_flag $time_flag $validate_flag \
        $device_flag --force_shift $FORCE_SHIFT \
        --log_dir "$log_dir" \
        $extra_args
}

# ============================================================================
cd "$SRC"
echo "============================================"
echo "  VLDB Distance Full Benchmark"
echo "  Started: $(date)"
echo "  16 baselines + Dist2GNN × 5 cities"
echo "============================================"

# ---- Phase 1: Non-ML ----
echo -e "\n${CYAN}===== Phase 1: Non-ML Baselines =====${NC}"
for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"
    run_exp "landmark" "$city" "$query" "--landmark_selection random --select_landmarks_from_train" "rn"
    run_exp "landmark" "$city" "$query" "--landmark_selection kmeans --select_landmarks_from_train" "km"
    run_exp "lpnorm"    "$city" "$query" "--p_norm 1" "manhattan"
done

# ---- Phase 2: NN Methods ----
echo -e "\n${CYAN}===== Phase 2: NN Methods =====${NC}"
for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"
    run_exp "geodnn"       "$city" "$query" "" ""
    run_exp "distancenn"   "$city" "$query" "" ""
    run_exp "embeddingnn"  "$city" "$query" "" ""
    run_exp "vdist2vec"    "$city" "$query" "" ""
    run_exp "ndist2vec"    "$city" "$query" "" ""
    run_exp "catboostnn"   "$city" "$query" "" ""
done

# ---- Phase 3: Functional Methods ----
echo -e "\n${CYAN}===== Phase 3: Functional Methods =====${NC}"
for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"
    run_exp "path2vec"  "$city" "$query" "" ""
    run_exp "aneda"     "$city" "$query" "" ""
    run_exp "rne"       "$city" "$query" "" ""
done

# ---- Phase 4: GNN Methods ----
echo -e "\n${CYAN}===== Phase 4: GNN Methods =====${NC}"
for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"
    run_exp "rgnndist2vec" "$city" "$query" "--gnn_layer gcn"  "gcn"
    run_exp "rgnndist2vec" "$city" "$query" "--gnn_layer sage" "sage"
    run_exp "rgnndist2vec" "$city" "$query" "--gnn_layer gat"  "gat"
    run_exp "dist2gnn"     "$city" "$query" \
        "--gnn_layer sage --dist2gnn_r 23 --dist2gnn_s 2 --dist2gnn_hidden 512 --dist2gnn_output 64 --dist2gnn_landmark_ratio 0.6" ""
done

# ---- Phase 5: GBDT ----
echo -e "\n${CYAN}===== Phase 5: GBDT =====${NC}"
for entry in "${CITIES_DATA[@]}"; do
    IFS='|' read -r city query <<< "$entry"
    run_exp "catboost" "$city" "$query" "" ""
done

ELAPSED=$(( ($(date +%s) - TOTAL_START) / 60 ))
echo -e "\n${GREEN}===== ALL DONE (${ELAPSED} min) =====${NC}"

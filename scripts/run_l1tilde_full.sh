#!/bin/bash
# L1Tilde ablation: 5 baselines × 5 cities = 25 experiments
# Replaces L1 with L1_tilde; compares against existing L1 results

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # l1tilde-metric-study/
SRC="${STUDY_DIR}"
PY="${PYTHON:-python3}"
cd $SRC

DEVICE="${DEVICE:-}"
[ -n "$DEVICE" ] && DEVICE_FLAG="--device $DEVICE" || DEVICE_FLAG=""
COMMON="$DEVICE_FLAG --force_shift 0 --epochs 20 --time_limit 5 --validate --l1tilde_r 62 --l1tilde_s 2 --embedding_dim 64"

run_exp() {
    local model=$1 city=$2 data_dir=$3 query_dir=$4 loss=$5 lr=$6 extra_args=$7
    local log_dir="results/${city}_${model}"
    echo "=== $(date '+%H:%M:%S')  $model  $city ==="
    $PY -u train.py \
        --model_class $model --data_dir $data_dir --query_dir $query_dir \
        --learning_rate $lr --loss $loss $COMMON $extra_args \
        --log_dir $log_dir 2>&1 | grep -E "Test MRE|MRE:|Finished|Error" | tail -3
    $PY -c "
import json
d=json.load(open('$log_dir/experiment_results.json'))
t=d['evaluation']['test']
print(f'  => Test MRE: {t[\"mre_percent\"]:.2f}%  |  Train MRE: {d[\"evaluation\"][\"train\"][\"mre_percent\"]:.2f}%')
"
    echo ""
}

# ===== Hrb_Small (small city, fast) =====
S="data/OSM_Harbin_Small"
Q="data/OSM_Harbin_Small/random_500k"
run_exp rgnndist2vec_l1tilde OSM_Harbin_Small "$S" "$Q" smoothl1 0.01 "--gnn_layer gat"
run_exp rgnndist2vec_l1tilde OSM_Harbin_Small "$S" "$Q" smoothl1 0.01 "--gnn_layer sage"
run_exp rgnndist2vec_l1tilde OSM_Harbin_Small "$S" "$Q" smoothl1 0.01 "--gnn_layer gcn"
run_exp rne_l1tilde OSM_Harbin_Small "$S" "$Q" mse 0.003 ""
run_exp lpnorm_l1tilde OSM_Harbin_Small "$S" "$Q" mse 0.01 "--epochs 1 --p_norm 1 --l1tilde_r 1 --l1tilde_s 1"

# ===== Large cities =====
for city in Harbin Chengdu Qingdao Beijing; do
    D="data/OSM_${city}"
    Q="data/OSM_${city}/proportional"
    run_exp rgnndist2vec_l1tilde "OSM_${city}" "$D" "$Q" smoothl1 0.01 "--gnn_layer gat"
    run_exp rgnndist2vec_l1tilde "OSM_${city}" "$D" "$Q" smoothl1 0.01 "--gnn_layer sage"
    run_exp rgnndist2vec_l1tilde "OSM_${city}" "$D" "$Q" smoothl1 0.01 "--gnn_layer gcn"
    run_exp rne_l1tilde "OSM_${city}" "$D" "$Q" mse 0.003 ""
    run_exp lpnorm_l1tilde "OSM_${city}" "$D" "$Q" mse 0.01 "--epochs 1 --p_norm 1 --l1tilde_r 1 --l1tilde_s 1"
done

echo "===== ALL DONE ====="
echo "Run: python3 /tmp/collect_l1tilde.py  to see results"

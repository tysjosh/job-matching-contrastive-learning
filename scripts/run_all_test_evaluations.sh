#!/bin/bash
# Run ordinal evaluation on TEST set for all 8 learning curve checkpoints

TEST_DATA="preprocess/data_splits_v6/test.jsonl"
FIXED_CFG="CDCL/config/phase1_ordinal_fixed_margin_config.json"
PHI_CFG="CDCL/config/phase1_ordinal_config.json"

run_eval() {
    DIR="$1"
    CFG="$2"
    CKPT="$DIR/phase1_pretraining/best_checkpoint.pt"
    OUT="$DIR/test_evaluation"
    echo "=========================================="
    echo "Evaluating: $DIR on TEST set"
    echo "=========================================="
    python run_ordinal_evaluation.py \
        --dataset "$TEST_DATA" \
        --ordinal-checkpoint "$CKPT" \
        --ordinal-config "$CFG" \
        --output-dir "$OUT"
    echo ""
}

run_eval "results_lc_10pct_fixed_margin" "$FIXED_CFG"
run_eval "results_lc_10pct_phi_corrected" "$PHI_CFG"
run_eval "results_lc_25pct_fixed_margin" "$FIXED_CFG"
run_eval "results_lc_25pct_phi_corrected" "$PHI_CFG"
run_eval "results_lc_50pct_fixed_margin" "$FIXED_CFG"
run_eval "results_lc_50pct_phi_corrected" "$PHI_CFG"
run_eval "results_ordinal_v6_fixed_margin" "$FIXED_CFG"
run_eval "results_ordinal_v6_phi_corrected" "$PHI_CFG"

echo "All test evaluations complete."

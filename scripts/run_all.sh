#!/usr/bin/env bash
#
# Run the full pipeline end-to-end.
#
# Usage:
#   bash scripts/run_all.sh [--model MODEL_NAME]
#
# Each experiment writes its results under experimental_outputs/ and
# updates results_summary.json. Downstream experiments read from the
# previous experiment's summary file as the source of truth.

set -euo pipefail

MODEL_ARG=""
if [[ "$#" -gt 0 ]]; then
    if [[ "$1" == "--model" && -n "${2:-}" ]]; then
        MODEL_ARG="--model $2"
        shift 2
    fi
fi

CONFIGS_DIR=${CONFIGS_DIR:-configs}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

echo "=== Experiment 1: Activation Patching ==="
python scripts/run_patching.py  --config "$CONFIGS_DIR/patching.yaml"  --log-file "$LOG_DIR/exp1_patching.log"  $MODEL_ARG

echo "=== Experiment 2: Truth Direction Optimization ==="
python scripts/run_tdo.py       --config "$CONFIGS_DIR/tdo.yaml"       --log-file "$LOG_DIR/exp2_tdo.log"       $MODEL_ARG

echo "=== Experiment 3: Truth Cone Optimization ==="
python scripts/run_tco.py       --config "$CONFIGS_DIR/tco.yaml"       --log-file "$LOG_DIR/exp3_tco.log"       $MODEL_ARG

echo "=== Experiment 4: DIM vs Cone Alignment (post-hoc, no GPU) ==="
python scripts/run_alignment.py --config "$CONFIGS_DIR/alignment.yaml" --log-file "$LOG_DIR/exp4_alignment.log"

echo "=== Experiment 5: Retention via KL ==="
python scripts/run_retention.py --config "$CONFIGS_DIR/retention.yaml" --log-file "$LOG_DIR/exp5_retention.log" $MODEL_ARG

echo
echo "All experiments finished. See experimental_outputs/ for artifacts."

#!/bin/bash
# Stage 0: Posthoc T sweep on fixed-999 model
# Purpose: Diagnose whether changing T at inference time has any benefit
#          when the model was trained with fixed T=999.
#
# This script calls the existing experiment_oracle_timestep.py with
# parameters appropriate for Stage 0 diagnostics.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_stage0_sweep.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_stage0_sweep.sh --max_images 24  # quick test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="./results/stage0_posthoc_sweep"

python src/experiment_oracle_timestep.py \
    --base_config ./configs/base.yaml \
    --test_config ./configs/test.yaml \
    --out_dir "$OUT_DIR" \
    --num_lambdas 6 \
    --lambda_min 0.2 \
    --lambda_max 128.0 \
    --t_min 800 \
    --t_max 999 \
    --t_step 25 \
    --t_ref 999 \
    --oracle_metric lpips \
    --make_plots \
    "$@"

echo ""
echo "Stage 0 sweep complete. Results in: $OUT_DIR"
echo "  - oracle_timestep_sweep.csv"
echo "  - oracle_timestep_summary.csv"
echo "  - correlation_table.csv"
echo ""
echo "Next: run diagnostic analysis:"
echo "  python src/stage0_analysis.py --sweep_dir $OUT_DIR"

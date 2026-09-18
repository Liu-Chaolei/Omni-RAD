#!/bin/bash
# Stage 2: Oracle-T sweep on T-augmentation model
# Purpose: Generate reliable per-image T_oracle_aug labels from a model
#          that has been trained with multiple timesteps (T-augmentation).
#
# This script calls the existing experiment_oracle_timestep.py with
# the T-aug checkpoint. Uses the same T candidates and lambda grid as
# Stage 0 for direct comparison.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_stage2_sweep.sh /path/to/t_aug_checkpoint.pth.tar
#   CUDA_VISIBLE_DEVICES=0 bash run_stage2_sweep.sh /path/to/ckpt.pth.tar --max_images 24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: bash run_stage2_sweep.sh <T-AUG_CHECKPOINT_PATH> [extra args...]"
    echo ""
    echo "Example:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_stage2_sweep.sh /data/results/t_aug/checkpoint.pth.tar"
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_stage2_sweep.sh /data/results/t_aug/checkpoint.pth.tar --max_images 24"
    exit 1
fi

CODEC_PATH="$1"
shift

if [ ! -f "$CODEC_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CODEC_PATH"
    exit 1
fi

OUT_DIR="./results/stage2_oracle_sweep"

echo "Stage 2: Oracle-T sweep on T-augmentation model"
echo "  Checkpoint: $CODEC_PATH"
echo "  Output dir: $OUT_DIR"
echo ""

python src/experiment_oracle_timestep.py \
    --base_config ./configs/base.yaml \
    --test_config ./configs/test.yaml \
    --codec_path "$CODEC_PATH" \
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
echo "Stage 2 sweep complete. Results in: $OUT_DIR"
echo "  - oracle_timestep_sweep.csv"
echo "  - oracle_timestep_summary.csv"
echo "  - correlation_table.csv"
echo ""
echo "Next: run diagnostic analysis (with Stage 0 cross-comparison):"
echo "  python src/stage2_analysis.py --sweep_dir $OUT_DIR --stage0_dir ./results/stage0_posthoc_sweep"

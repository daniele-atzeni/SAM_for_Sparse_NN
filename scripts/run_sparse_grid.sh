#!/usr/bin/env bash
# Full SAM-vs-SGD-under-pruning grid: {ResNet18, VGG16} x {s=0.7, s=0.9} x 3 seeds.
# Each invocation of main_training_sparse.py trains BOTH SAM and SGD from the
# same random init for that (arch, sparsity, seed) cell (fair comparison).
#
# Checkpoint policy: FULL_CKPT_SEED keeps a checkpoint every 5 epochs
# (finer than the 10-epoch pruning cadence, so it also catches mid-recovery
# state between rounds); the other two seeds only keep the final model, to
# avoid filling the disk with near-duplicate checkpoints across seeds.
#
# Usage:
#   bash scripts/run_sparse_grid.sh
#   bash scripts/run_sparse_grid.sh 13        # just seed 13 (with full checkpoints)
#   bash scripts/run_sparse_grid.sh 42 97     # just seeds 42 and 97 (final-only)

set -euo pipefail
cd "$(dirname "$0")/.."

FULL_CKPT_SEED=13
FULL_CKPT_SAVE_EVERY=5

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

CONFIGS=(
  "configs/sparse/ResNet18_CIFAR10_s0.7.json"
  "configs/sparse/ResNet18_CIFAR10_s0.9.json"
  "configs/sparse/VGG16_CIFAR10_s0.7.json"
  "configs/sparse/VGG16_CIFAR10_s0.9.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse: $config | seed $seed | log: $log_file ==="
    if [ "$seed" -eq "$FULL_CKPT_SEED" ]; then
      python main_training_sparse.py --config "$config" --seed "$seed" \
        --save-every "$FULL_CKPT_SAVE_EVERY" \
        2>&1 | tee "$log_file"
    else
      python main_training_sparse.py --config "$config" --seed "$seed" \
        2>&1 | tee "$log_file"
    fi
  done
done

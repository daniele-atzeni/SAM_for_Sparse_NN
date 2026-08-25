#!/usr/bin/env bash
# Stronger-pruning follow-up to run_sparse_grid.sh: {ResNet18, VGG16} x
# {s=0.95, s=0.98, s=0.99} x 3 seeds, same schedule (first_iter=15,
# prune_every=10, n_iter=5, 180 epochs -- pruning still finishes at epoch 55,
# 125 epochs of recovery). This isolates sparsity as the only changed
# variable vs. the s=0.7/0.9 grid, to see whether pushing sparsity alone
# (holding the recovery budget fixed) produces visible SAM-vs-SGD /
# dense-vs-sparse divergence in the training trajectories, not just the
# final numbers.
#
# Checkpoint policy: same as run_sparse_grid.sh -- FULL_CKPT_SEED keeps a
# checkpoint every 5 epochs, the other two seeds keep only the final model.
#
# Usage:
#   bash scripts/run_sparse_grid_strong.sh
#   bash scripts/run_sparse_grid_strong.sh 13        # just seed 13
#   bash scripts/run_sparse_grid_strong.sh 42 97     # just seeds 42 and 97

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
  "configs/sparse/ResNet18_CIFAR10_s0.95.json"
  "configs/sparse/ResNet18_CIFAR10_s0.98.json"
  "configs/sparse/ResNet18_CIFAR10_s0.99.json"
  "configs/sparse/VGG16_CIFAR10_s0.95.json"
  "configs/sparse/VGG16_CIFAR10_s0.98.json"
  "configs/sparse/VGG16_CIFAR10_s0.99.json"
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

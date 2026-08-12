#!/usr/bin/env bash
# Full SAM-vs-SGD-under-pruning grid: {ResNet18, VGG16} x {s=0.7, s=0.9} x 3 seeds.
# Each invocation of main_training_sparse.py trains BOTH SAM and SGD from the
# same random init for that (arch, sparsity, seed) cell (fair comparison).
#
# Checkpoint policy: seed 0 keeps a checkpoint at every pruning round
# (--save-every = prune_every from the configs, currently 10); seeds 1 and 2
# only keep the final model, to avoid filling the disk with near-duplicate
# checkpoints across seeds. Adjust FULL_CKPT_SAVE_EVERY below if you change
# prune_every in the configs.
#
# Usage:
#   bash scripts/run_sparse_grid.sh
#   bash scripts/run_sparse_grid.sh 0        # just seed 0 (with full checkpoints)
#   bash scripts/run_sparse_grid.sh 1 2      # just seeds 1 and 2 (final-only)

set -euo pipefail
cd "$(dirname "$0")/.."

FULL_CKPT_SEED=0
FULL_CKPT_SAVE_EVERY=10

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 1 2)
fi

CONFIGS=(
  "configs/sparse/ResNet18_CIFAR10_s0.7.json"
  "configs/sparse/ResNet18_CIFAR10_s0.9.json"
  "configs/sparse/VGG16_CIFAR10_s0.7.json"
  "configs/sparse/VGG16_CIFAR10_s0.9.json"
)

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== sparse: $config | seed $seed ==="
    if [ "$seed" -eq "$FULL_CKPT_SEED" ]; then
      python main_training_sparse.py --config "$config" --seed "$seed" \
        --save-every "$FULL_CKPT_SAVE_EVERY"
    else
      python main_training_sparse.py --config "$config" --seed "$seed"
    fi
  done
done

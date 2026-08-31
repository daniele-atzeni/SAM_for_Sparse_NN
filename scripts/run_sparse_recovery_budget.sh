#!/usr/bin/env bash
# Recovery-budget follow-up: {ResNet18, VGG16} x s=0.9, same architectures
# and target sparsity as the original grid, but with a much shorter recovery
# window -- pruning continues every 15 epochs for 11 rounds (epochs 15, 30,
# ..., 165), reaching 0.9 sparsity only 15 epochs before the end of training,
# instead of the original schedule's 5 rounds finishing at epoch 55 (125
# epochs of recovery). s=0.9 was chosen because it already showed no final-
# accuracy gap between SAM/SGD under the generous-recovery schedule -- this
# isolates the recovery-budget variable while holding sparsity fixed, to
# check whether a short recovery window is what's needed to see accuracy
# itself (not just curvature/loss trajectory) diverge between optimizers.
#
# Checkpoint policy: same as the other sparse scripts -- FULL_CKPT_SEED
# keeps a checkpoint every 5 epochs, the other two seeds keep only the
# final model.
#
# Usage:
#   bash scripts/run_sparse_recovery_budget.sh
#   bash scripts/run_sparse_recovery_budget.sh 13
#   bash scripts/run_sparse_recovery_budget.sh 42 97

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
  "configs/sparse/ResNet18_CIFAR10_s0.9_shortrecovery.json"
  "configs/sparse/VGG16_CIFAR10_s0.9_shortrecovery.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (short recovery): $config | seed $seed | log: $log_file ==="
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

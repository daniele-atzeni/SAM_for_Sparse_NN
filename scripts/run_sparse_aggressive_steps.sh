#!/usr/bin/env bash
# Isolates cut *size* from recovery *budget*. scripts/run_sparse_recovery_budget.sh
# changed two things vs. the original grid at once: more rounds (11 vs 5)
# AND less final recovery (15 vs 125 epochs) -- and more rounds at the same
# target sparsity means SMALLER individual cuts (~17.8% of remaining weights
# per round at n_iter=11, vs ~36.9% at the original n_iter=5). So on "how
# brutal is each cut" that run was actually the gentlest tested, not the
# most aggressive.
#
# This config holds the exact same total window as the short-recovery run
# (first_iter=15, ends epoch 165, 15 epochs of final recovery) but uses only
# 3 rounds instead of 11 -- each one removing ~53.6% of remaining weights.
# Same target sparsity (0.9), same total recovery budget, only the cut size
# changed. Directly comparable to run_sparse_recovery_budget.sh's results.
#
# Checkpoint policy: same as the other sparse scripts -- FULL_CKPT_SEED
# keeps a checkpoint every 5 epochs, the other two seeds keep only the
# final model.
#
# Usage:
#   bash scripts/run_sparse_aggressive_steps.sh
#   bash scripts/run_sparse_aggressive_steps.sh 13
#   bash scripts/run_sparse_aggressive_steps.sh 42 97

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
  "configs/sparse/ResNet18_CIFAR10_s0.9_aggressivesteps.json"
  "configs/sparse/VGG16_CIFAR10_s0.9_aggressivesteps.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (aggressive steps): $config | seed $seed | log: $log_file ==="
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

#!/usr/bin/env bash
# Final experiment: does the short-recovery SAM advantage replicate on a
# harder dataset? Same architectures, same s=0.9, same short-recovery
# schedule (11 rounds every 15 epochs, finishing at epoch 165, 15 epochs of
# recovery) as scripts/run_sparse_recovery_budget.sh -- the ONE condition
# that showed a consistent final-accuracy gap on CIFAR-10 (+0.9pp mean,
# SAM ahead in 5/6 seed x architecture cells). CIFAR-100 has the same
# 50,000 training images but spread across 100 classes instead of 10 (500
# images/class instead of 5,000), and a much lower achievable ceiling --
# less slack for either optimizer to recover into, so if the effect is
# real and recovery-budget-dependent, this is where it should show up
# more clearly, not less.
#
# Checkpoint policy: same as the other sparse scripts -- FULL_CKPT_SEED
# keeps a checkpoint every 5 epochs, the other two seeds keep only the
# final model.
#
# Usage:
#   bash scripts/run_sparse_cifar100.sh
#   bash scripts/run_sparse_cifar100.sh 13
#   bash scripts/run_sparse_cifar100.sh 42 97

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
  "configs/sparse/ResNet18_CIFAR100_s0.9_shortrecovery.json"
  "configs/sparse/VGG16_CIFAR100_s0.9_shortrecovery.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (CIFAR-100, short recovery): $config | seed $seed | log: $log_file ==="
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

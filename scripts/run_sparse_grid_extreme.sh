#!/usr/bin/env bash
# Capacity-wall follow-up: {ResNet18, VGG16} x {s=0.995, 0.999, 0.9995} --
# same schedule as the original grid
# (first_iter=15, prune_every=10, n_iter=5, 125 epochs of recovery), pushing
# sparsity far beyond the s=0.7-0.99 sweep, which showed SGD still reaching
# ~99-100% *training* accuracy at every sparsity tested -- i.e. the network
# was still comfortably over-parameterized (able to fully memorize the
# 50,000-image training set) even at s=0.99 (~112k active weights for
# ResNet18, ~336k for VGG16). This sweep targets active-parameter counts
# down near/below the training-set size itself:
#   ResNet18 (11,164,352 total): s=0.995 -> ~55.8k active, s=0.999 -> ~11.2k,
#     s=0.9995 -> ~5.6k active
#   VGG16 (33,625,792 total): s=0.995 -> ~168.1k active, s=0.999 -> ~33.6k,
#     s=0.9995 -> ~16.8k active
# The recovery budget is held fixed (same as the original grid) so sparsity
# is the only changed variable -- the goal is to find where *SGD's* training
# accuracy actually starts dropping well below 100%, which is the direct
# empirical signature of the under-parameterized regime, as opposed to a
# curvature/trajectory difference that still fully recovers.
#
# Checkpoint policy: same as the other sparse scripts -- FULL_CKPT_SEED
# keeps a checkpoint every 5 epochs, the other two seeds keep only the
# final model.
#
# Usage:
#   bash scripts/run_sparse_grid_extreme.sh
#   bash scripts/run_sparse_grid_extreme.sh 13
#   bash scripts/run_sparse_grid_extreme.sh 42 97

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
  "configs/sparse/ResNet18_CIFAR10_s0.995.json"
  "configs/sparse/ResNet18_CIFAR10_s0.999.json"
  "configs/sparse/ResNet18_CIFAR10_s0.9995.json"
  "configs/sparse/VGG16_CIFAR10_s0.995.json"
  "configs/sparse/VGG16_CIFAR10_s0.999.json"
  "configs/sparse/VGG16_CIFAR10_s0.9995.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (extreme sparsity): $config | seed $seed | log: $log_file ==="
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

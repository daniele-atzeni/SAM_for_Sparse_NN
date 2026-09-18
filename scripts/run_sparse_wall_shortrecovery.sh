#!/usr/bin/env bash
# Does the capacity wall move when recovery is tight at every round, not
# just the last one? Reuses the proven short-recovery schedule shape (11
# rounds every 15 epochs, finishing at epoch 165, 15 epochs of recovery
# after EVERY round, not only the final one -- unlike the 3-round
# aggressivesteps variant, which left 75 epochs between rounds 1-2 and
# 2-3 and showed no visible dip at all) but now targets sparsities at and
# around each architecture's known capacity wall under generous recovery:
#   ResNet18: s=0.995 (fine, ~96% train acc under generous recovery) and
#     s=0.999 (collapsed, ~84% train acc under generous recovery)
#   VGG16:    s=0.999 (fine, ~90% train acc under generous recovery) and
#     s=0.9995 (collapsed, ~77% train acc under generous recovery)
# If tight recovery moves the wall to a lower sparsity than under generous
# recovery, the "fine" configs here should show real degradation (and a
# real, visible round-by-round gap) that they didn't show before.
#
# These configs reuse prune_ratio values from the already-completed
# extreme-sparsity sweep (run_sparse_grid_extreme.sh). That's safe now --
# main_training_sparse.py tags saved_models/ and tensorboard/ output by
# the config file's own name, not just prune_ratio, so this run cannot
# collide with or overwrite that sweep's checkpoints. (It didn't used to
# be safe: the aggressive-steps and short-recovery s=0.9 runs silently
# overwrote each other's and the original baseline's checkpoints before
# this was fixed, since all three shared prune_ratio=0.9.)
#
# Checkpoint policy: same as the other sparse scripts -- FULL_CKPT_SEED
# keeps a checkpoint every 5 epochs, the other two seeds keep only the
# final model.
#
# Usage:
#   bash scripts/run_sparse_wall_shortrecovery.sh
#   bash scripts/run_sparse_wall_shortrecovery.sh 13
#   bash scripts/run_sparse_wall_shortrecovery.sh 42 97

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
  "configs/sparse/ResNet18_CIFAR10_s0.995_shortrecovery.json"
  "configs/sparse/ResNet18_CIFAR10_s0.999_shortrecovery.json"
  "configs/sparse/VGG16_CIFAR10_s0.999_shortrecovery.json"
  "configs/sparse/VGG16_CIFAR10_s0.9995_shortrecovery.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (wall x short recovery): $config | seed $seed | log: $log_file ==="
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

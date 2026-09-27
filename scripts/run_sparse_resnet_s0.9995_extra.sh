#!/usr/bin/env bash
# One extra wall x short-recovery data point: ResNet18 at s=0.9995, one step
# past the s=0.999 wall config already in the sweep (roughly halves active
# params again: ~11.2k -> ~5.6k). Confirmed non-degenerate first -- the
# earlier generous-recovery capacity-wall sweep already ran ResNet18 at this
# exact sparsity and got 76-80% test accuracy (seeds 13/42/97), so this
# isn't a shot in the dark, just untested under tight recovery.
#
# Purpose: does the SAM-SGD gap (1.4pp at s=0.995 -> 4.1pp at s=0.999) keep
# growing at s=0.9995, or saturate/reverse as both optimizers approach a
# shared capacity floor? Also gives the SAM reference number and the
# "normal" (epoch-matched) gap to compare against the iso-compute SGD run
# at the same sparsity (configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd.json,
# added to scripts/run_sparse_isocompute_sgd.sh).
#
# Usage:
#   bash scripts/run_sparse_resnet_s0.9995_extra.sh
#   bash scripts/run_sparse_resnet_s0.9995_extra.sh 13

set -euo pipefail
cd "$(dirname "$0")/.."

FULL_CKPT_SEED=13
FULL_CKPT_SAVE_EVERY=5

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

config="configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery.json"
name="$(basename "$config" .json)"
log_dir="logs/sparse/$name"
mkdir -p "$log_dir"
for seed in "${SEEDS[@]}"; do
  log_file="$log_dir/seed_${seed}.log"
  echo "=== sparse (extra wall point): $config | seed $seed | log: $log_file ==="
  if [ "$seed" -eq "$FULL_CKPT_SEED" ]; then
    python main_training_sparse.py --config "$config" --seed "$seed" \
      --save-every "$FULL_CKPT_SAVE_EVERY" \
      2>&1 | tee "$log_file"
  else
    python main_training_sparse.py --config "$config" --seed "$seed" \
      2>&1 | tee "$log_file"
  fi
done

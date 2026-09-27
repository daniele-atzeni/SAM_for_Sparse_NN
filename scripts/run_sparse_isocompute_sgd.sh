#!/usr/bin/env bash
# The iso-compute check: does SGD close the gap if given the SAME TOTAL
# TRAINING FLOPS as SAM, instead of the same epoch count?
#
# SAM does 2 full forward+backward passes per step (see
# src/train/training.py: loss.backward(); first_step; model(data) again;
# sam_loss.backward(); second_step) -- confirmed by direct measurement
# (torch.utils.flop_counter) at 2.000x SGD's per-step FLOPs for both
# architectures. Pruning itself (unstructured masking via
# prune.global_unstructured) does NOT reduce FLOPs -- the mask zeroes a
# dense tensor but every matmul still runs at full size -- so at a fixed
# epoch count, sparse and dense cost the same, and the entire compute
# asymmetry in the whole campaign is SAM's 2x per step. See
# figures/compute_vs_accuracy.png / TODO.md for the full breakdown.
#
# Scoped to ResNet18 only, two sparsities, per the priority call: at
# s=0.995 both optimizers' post-cut recovery curves are already flat by
# epoch 180 (mean slope ~0.03-0.08pp/epoch over the last 5 epochs), so more
# time wouldn't be expected to move that small a gap. At s=0.999 SGD is
# still visibly climbing when training stops (~0.26pp/epoch vs SAM's
# ~0.08pp/epoch over the same window) -- the gap there is measured before
# either curve has converged, so this is the sparsity where the answer is
# genuinely open. s=0.9995 (~5.6k active params, half of s=0.999's ~11.2k)
# adds a second point to see whether the same still-climbing pattern (and
# the underlying accuracy gap) continues, saturates, or reverses one step
# further past the capacity wall -- confirmed non-degenerate first (76-80%
# test accuracy under generous recovery, seeds 13/42/97, from the earlier
# capacity-wall sweep). VGG16 is intentionally left out for now (lower
# priority; add configs/sparse/VGG16_CIFAR10_s0.9995_shortrecovery_isocompute_sgd.json
# back into CONFIGS below if full architecture coverage becomes worth it).
#
# Both configs give SGD 360 epochs (2x the original 180), so total SGD
# FLOPs = total SAM FLOPs from the original 180-epoch runs. The pruning
# schedule shape is preserved (11 rounds, same final sparsity) by doubling
# both prune_every and first_iter (15->30), and the LR schedule is scaled
# proportionally (milestones at the same 44%/89% fraction of training) so
# the back half of training isn't wasted at a decayed LR.
#
# Only SGD is run here (--use-sam False) -- the SAM reference numbers
# already exist from the 180-epoch wall x short-recovery sweep (s=0.999)
# and from scripts/run_sparse_resnet_s0.9995_extra.sh (s=0.9995, run that
# first), and running SAM for another 360-epoch pass isn't needed to
# answer "does equal-compute SGD catch up to SAM."
#
# Usage:
#   bash scripts/run_sparse_isocompute_sgd.sh
#   bash scripts/run_sparse_isocompute_sgd.sh 13
#   bash scripts/run_sparse_isocompute_sgd.sh 42 97

set -euo pipefail
cd "$(dirname "$0")/.."

FULL_CKPT_SEED=13
FULL_CKPT_SAVE_EVERY=10

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

CONFIGS=(
  "configs/sparse/ResNet18_CIFAR10_s0.999_shortrecovery_isocompute_sgd.json"
  "configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd.json"
  # "configs/sparse/VGG16_CIFAR10_s0.9995_shortrecovery_isocompute_sgd.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== sparse (iso-compute SGD): $config | seed $seed | log: $log_file ==="
    if [ "$seed" -eq "$FULL_CKPT_SEED" ]; then
      python main_training_sparse.py --config "$config" --seed "$seed" \
        --use-sam False --save-every "$FULL_CKPT_SAVE_EVERY" \
        2>&1 | tee "$log_file"
    else
      python main_training_sparse.py --config "$config" --seed "$seed" \
        --use-sam False \
        2>&1 | tee "$log_file"
    fi
  done
done

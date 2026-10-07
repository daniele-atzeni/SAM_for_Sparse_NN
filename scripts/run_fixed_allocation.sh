#!/usr/bin/env bash
# Fixed per-layer allocation control (ResNet-18, CIFAR-10, s=0.9995,
# constrained iterative schedule). Every run, SAM and SGD, all seeds, prunes
# each layer to the per-round allocation that global magnitude pruning
# produced for SGD (seed 13), keeping the largest weights within each layer.
# One stream per seed: SAM first, then SGD. Curvature diagnostics disabled.
#
# Usage:
#   bash scripts/run_fixed_allocation.sh
set -euo pipefail
cd "$(dirname "$0")/.."

config="configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_fixedalloc.json"
log_dir="logs/sparse/$(basename "$config" .json)"
mkdir -p "$log_dir"
for seed in 13 42 97; do
  (
    for sam in True False; do
      python main_training_sparse.py --config "$config" --seed "$seed" --use-sam "$sam" \
        > "$log_dir/seed_${seed}_sam_${sam}.log" 2>&1
    done
  ) &
done
wait

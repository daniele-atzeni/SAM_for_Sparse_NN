#!/usr/bin/env bash
# rho sweep for the headline setting (ResNet-18, CIFAR-10, s=0.9995,
# constrained iterative pruning). SAM only: the SGD runs at the same
# schedule already exist. One background stream per rho value, each looping
# over the 3 seeds. Curvature diagnostics are disabled
# (evaluate_flatness_every=1000) so a run costs only training + accuracy.
#
# Usage:
#   bash scripts/run_rho_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for rho in 0.05 0.1 0.2; do
  config="configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_rho${rho}.json"
  name="$(basename "$config" .json)"
  log_dir="logs/sparse/$name"
  mkdir -p "$log_dir"
  (
    for seed in 13 42 97; do
      python main_training_sparse.py --config "$config" --seed "$seed" --use-sam True \
        > "$log_dir/seed_${seed}.log" 2>&1
    done
  ) &
done
wait

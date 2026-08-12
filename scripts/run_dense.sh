#!/usr/bin/env bash
# Optional dense (no pruning) baselines for ResNet18 and VGG16 on CIFAR-10.
# Not required for the SAM-vs-SGD-under-pruning comparison itself (the sparse
# runs prune from a shared random init, not from a dense-trained model) --
# useful only as a reference "how good could this architecture get" number.
#
# Usage:
#   bash scripts/run_dense.sh [seeds...]
#   bash scripts/run_dense.sh            # defaults to seeds 13 42 97
#   bash scripts/run_dense.sh 13          # just seed 13

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

ARCH_CONFIGS=(
  "configs/dense/ResNet18_CIFAR10.json"
  "configs/dense/VGG16_CIFAR10.json"
)

for config in "${ARCH_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== dense: $config | seed $seed ==="
    python main_training_dense.py --config "$config" --seed "$seed"
  done
done

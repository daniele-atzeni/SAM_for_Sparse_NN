#!/usr/bin/env bash
# CNN-side validation of the three-factor decomposition (Proposition 3.2),
# the one still missing from the paper (pPRQ Q1, meta-review point 4) --
# Table 1 only covers MLPs. Reuses the existing dense checkpoints, no new
# training needed. Cheap: a handful of backward passes per model, no
# Hessian power iteration.
#
# Usage:
#   bash scripts/run_three_factor_cnn.sh
#   bash scripts/run_three_factor_cnn.sh 13

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

CONFIGS=("configs/dense/ResNet18_CIFAR10.json" "configs/dense/VGG16_CIFAR10.json")
TAGS=("ResNet18_cifar10" "vgg16_bn_cifar10")

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  tag="${TAGS[$i]}"
  log_dir="logs/three_factor/$tag"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== three-factor CNN: $tag | seed $seed | log: $log_file ==="
    python scripts/analyze_three_factor_cnn.py \
      --dense-config "$config" --dense-tag "$tag" --seed "$seed" \
      2>&1 | tee "$log_file"
  done
done

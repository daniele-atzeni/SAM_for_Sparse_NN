#!/usr/bin/env bash
# Prune-then-finetune regime, revived with the pruned=True fix applied to
# train_loop (see main_prune_finetune.py docstring) -- this replaces the
# NeurIPS submission's Tables 10-12, which were generated pre-fix.
#
# Starts from the existing dense checkpoints (saved_models/dense/{dense_tag}/
# seed_{seed}/..._sam_{True,False}.pth, already trained for 180 epochs, both
# optimizers, all 3 seeds) -- so this is genuinely fast: just prune + 5
# epochs of finetuning, all 4 sam_train x sam_finetune combos, per
# pruning_ratio, per seed. No 180-epoch retrain needed.
#
# Matches the old paper's sparsity levels (0.5, 0.7, 0.9) and finetune
# duration (5 epochs) for a like-for-like table refresh.
#
# Usage:
#   bash scripts/run_prune_finetune.sh
#   bash scripts/run_prune_finetune.sh 13
#   bash scripts/run_prune_finetune.sh 42 97

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(13 42 97)
fi

CONFIGS=(
  "configs/finetune/ResNet18_CIFAR10.json"
  "configs/finetune/VGG16_CIFAR10.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log_dir="logs/finetune/$name"
  mkdir -p "$log_dir"
  for seed in "${SEEDS[@]}"; do
    log_file="$log_dir/seed_${seed}.log"
    echo "=== prune-finetune: $config | seed $seed | log: $log_file ==="
    python main_prune_finetune.py --config "$config" --seed "$seed" \
      2>&1 | tee "$log_file"
  done
done

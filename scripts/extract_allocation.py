"""Extract the per-round, per-layer allocation that global magnitude pruning
produced in one run, for use as a fixed allocation (config key
"allocation_file") in main_training_sparse.py.

The checkpoint at a cut epoch holds the weights at the end of that epoch,
with pruned weights stored as exact zeros, so its nonzero pattern is the
mask after that cut. Layers are listed in model.named_modules() order,
matching the Conv2d/Linear order used by the training loop.

Usage (server):
    python scripts/extract_allocation.py \
        --ckpt-dir saved_models/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery/seed_13/checkpoint \
        --sam False --out configs/allocations/resnet18_s0.9995_sgd_seed13.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

CUTS = list(range(15, 166, 15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--sam", default="False")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rounds = []
    for c in CUTS:
        state = torch.load(os.path.join(args.ckpt_dir, f"sam_{args.sam}_epoch_{c}.pth"), map_location="cpu")
        keys = [k for k, v in state.items() if k.endswith("weight") and v.dim() in (2, 4)]
        rounds.append([int((state[k] != 0).sum()) for k in keys])
        print(c, sum(rounds[-1]))
    json.dump({"source": args.ckpt_dir, "sam": args.sam, "cuts": CUTS, "layers": keys, "rounds": rounds},
              open(args.out, "w"), indent=1)
    print("Saved", args.out)


if __name__ == "__main__":
    main()

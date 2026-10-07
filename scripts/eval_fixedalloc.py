"""Full-test accuracy and effective sparsity of the fixed-allocation runs
(ResNet-18, s = 0.9995, every optimizer given SGD seed 13's per-layer counts).

Usage (server):
    python scripts/eval_fixedalloc.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_review4 import load_sparse, full_test_acc, masks_of, effective_masks
from src.registry import build_dataloaders

_, tl = build_dataloaders("cifar10", 512)
out = {}
for seed in [13, 42, 97]:
    for sam in ["True", "False"]:
        path = f"saved_models/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_fixedalloc/seed_{seed}/ResNet18_cifar10_sam_{sam}.pth"
        if not os.path.exists(path):
            continue
        m = load_sparse("ResNet18", path)
        masks = masks_of(m)
        eff = effective_masks("ResNet18", masks)
        key = f"seed_{seed}/SAM_{sam}"
        out[key] = {"acc": full_test_acc(m, tl),
                    "nominal": int(sum(v.sum().item() for v in masks.values())),
                    "effective": int(sum(v.sum().item() for v in eff.values())),
                    "per_layer_nominal": {n: int(v.sum().item()) for n, v in masks.items()}}
        print(key, out[key]["acc"], out[key]["nominal"], out[key]["effective"], flush=True)
json.dump(out, open("results/fixedalloc_fulltest.json", "w"), indent=1)

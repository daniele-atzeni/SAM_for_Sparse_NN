"""Fourth round of analyses on existing checkpoints (no training).

1. Per-layer pruned energy ||delta_l||^2 for the dense checkpoints at
   s = 0.5, 0.7, 0.9 (global magnitude threshold), so that a layer-wise
   generic-alignment prediction sum_l (tr_l / p_l) ||delta_l||^2 can be
   compared with the measured ||Delta f||^2 (per-layer traces come from
   results/existing_checkpoints_analysis.json, same batch and probes).
2. Norm of the active prunable weights ||theta_active|| over training for
   the fully checkpointed seed-13 iterative runs, hence rho / ||theta_active||.
3. Per-layer weight norms ||w_l|| at every checkpoint of those runs (to
   look at radial, BatchNorm-scale dynamics within the retraining windows).

Usage (server):
    python scripts/analyze_checkpoints_round4.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

SEEDS = [13, 42, 97]
DENSE = ["ResNet18_cifar10", "vgg16_bn_cifar10"]
SPARSITIES = [0.5, 0.7, 0.9]
TRAJ_RUNS = {
    "ResNet18_CIFAR10_s0.995_shortrecovery": "ResNet18_cifar10",
    "ResNet18_CIFAR10_s0.999_shortrecovery": "ResNet18_cifar10",
    "ResNet18_CIFAR10_s0.9995_shortrecovery": "ResNet18_cifar10",
    "VGG16_CIFAR10_s0.999_shortrecovery": "vgg16_bn_cifar10",
    "VGG16_CIFAR10_s0.9995_shortrecovery": "vgg16_bn_cifar10",
}


def prunable_keys(state):
    # Conv/Linear weights: 4-D conv kernels and 2-D linear matrices.
    return [k for k, v in state.items() if k.endswith("weight") and v.dim() in (2, 4)]


def main():
    out = {"layer_delta": {}, "trajectory": {}}

    for tag in DENSE:
        out["layer_delta"][tag] = {}
        for seed in SEEDS:
            for sam in ["True", "False"]:
                state = torch.load(f"saved_models/dense/{tag}/seed_{seed}/{tag}_sam_{sam}.pth", map_location="cpu")
                keys = prunable_keys(state)
                flat = torch.cat([state[k].abs().flatten() for k in keys])
                entry = {"keys": keys}
                for s in SPARSITIES:
                    thr = flat.kthvalue(int(s * flat.numel())).values.item()
                    entry[str(s)] = [state[k][state[k].abs() <= thr].pow(2).sum().item() for k in keys]
                out["layer_delta"][tag][f"{seed}|{sam}"] = entry
                print(f"delta {tag} seed={seed} SAM={sam}", flush=True)

    for run in TRAJ_RUNS:
        d = f"saved_models/sparse/{run}/seed_13/checkpoint"
        if not os.path.isdir(d):
            continue
        out["trajectory"][run] = {}
        for sam in ["True", "False"]:
            rows = []
            files = sorted(glob.glob(f"{d}/sam_{sam}_epoch_*.pth"),
                           key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)))
            for f in files:
                state = torch.load(f, map_location="cpu")
                keys = prunable_keys(state)
                norms = [state[k].norm().item() for k in keys]
                active = sum(int((state[k] != 0).sum()) for k in keys)
                rows.append({"epoch": int(re.search(r"epoch_(\d+)", f).group(1)),
                             "active": active,
                             "theta_active_norm": sum(n * n for n in norms) ** 0.5,
                             "layer_norms": norms})
            out["trajectory"][run][sam] = {"keys": keys, "rows": rows}
            print(f"trajectory {run} SAM={sam}: {len(rows)} checkpoints", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/checkpoints_round4.json", "w") as f:
        json.dump(out, f)
    print("Saved results/checkpoints_round4.json")


if __name__ == "__main__":
    main()

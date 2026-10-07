"""Third round of analyses on existing checkpoints (no training).

1. Layer allocation: per-layer density of the final iterative-pruning models,
   SAM vs SGD (global magnitude pruning lets each optimizer's weight scales
   decide how many weights each layer keeps).
2. BatchNorm staleness: full-test accuracy of the final models as trained
   (running statistics from training) and after recomputing BatchNorm
   statistics on 50 training batches.
3. Dense full-test accuracy of the dense SAM/SGD checkpoints.

Usage (server):
    python scripts/analyze_checkpoints_round3.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.data.cifar10 import load as load_cifar10
from src.registry import build_model

SEEDS = [13, 42, 97]
SPARSE_RUNS = [
    "ResNet18_CIFAR10_s0.995_shortrecovery",
    "ResNet18_CIFAR10_s0.999_shortrecovery",
    "ResNet18_CIFAR10_s0.9995_shortrecovery",
    "VGG16_CIFAR10_s0.999_shortrecovery",
    "VGG16_CIFAR10_s0.9995_shortrecovery",
]
DENSE = [("configs/dense/ResNet18_CIFAR10.json", "ResNet18_cifar10"),
         ("configs/dense/VGG16_CIFAR10.json", "vgg16_bn_cifar10")]
BN_BATCHES = 50


def load_any(name, params, path, device):
    model = build_model(name, params).to(device)
    state = torch.load(path, map_location=device)
    if any(k.endswith("weight_orig") for k in state):
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                prune.identity(m, "weight")
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
        n += y.numel()
    return correct / n


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= BN_BATCHES:
            break
        model(x.to(device))
    model.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, testset = load_cifar10(randomize=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2,
                                               generator=torch.Generator().manual_seed(0))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    out = {"dense_acc": {}, "sparse": {}}

    for cfg_path, tag in DENSE:
        cfg = json.load(open(cfg_path))
        out["dense_acc"][tag] = {}
        for seed in SEEDS:
            for sam in ["True", "False"]:
                model = load_any(cfg["model"]["name"], cfg["model"]["parameters"],
                                 f"saved_models/dense/{tag}/seed_{seed}/{tag}_sam_{sam}.pth", device)
                acc = accuracy(model, test_loader, device)
                out["dense_acc"][tag][f"{seed}|{sam}"] = acc
                print(f"dense {tag} seed={seed} SAM={sam}: {acc:.4f}", flush=True)

    for run in SPARSE_RUNS:
        cfg = json.load(open(f"configs/sparse/{run}.json"))
        name, params = cfg["model"]["name"], cfg["model"]["parameters"]
        tag = "ResNet18_cifar10" if name == "ResNet18" else "vgg16_bn_cifar10"
        out["sparse"][run] = {}
        for seed in SEEDS:
            for sam in ["True", "False"]:
                path = f"saved_models/sparse/{run}/seed_{seed}/{tag}_sam_{sam}.pth"
                state = torch.load(path, map_location="cpu")
                dens = {k[: -len(".weight_mask")]: v.float().mean().item()
                        for k, v in state.items() if k.endswith("weight_mask")}
                model = load_any(name, params, path, device)
                acc = accuracy(model, test_loader, device)
                recalibrate_bn(model, train_loader, device)
                acc_bn = accuracy(model, test_loader, device)
                out["sparse"][run][f"{seed}|{sam}"] = {"acc": acc, "acc_bn_recal": acc_bn, "layer_density": dens}
                print(f"{run} seed={seed} SAM={sam}: acc={acc:.4f} acc_bn_recal={acc_bn:.4f}", flush=True)
                del model

    os.makedirs("results", exist_ok=True)
    with open("results/checkpoints_round3.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved results/checkpoints_round3.json")


if __name__ == "__main__":
    main()

"""Analyses on checkpoints that already exist -- no training.

1. Full-test-set accuracy of the final iterative-pruning models (and the
   compute-matched SGD models). Training logs evaluate on the first
   eval_batches=10 test batches only (1,280 images); this re-evaluates every
   final model on all 10,000 test images and on that same 1,280-image
   subset, so the two can be compared directly.

2. Scale-invariant three-factor quantities for the dense checkpoints.
   With BatchNorm after a conv layer, rescaling its weights by c leaves the
   function unchanged but multiplies the layer's Jacobian trace by 1/c^2 and
   its pruned energy by c^2, so R_s and R_delta are not invariant. Per layer
   l we therefore also report
       s_tilde = sum_l ||w_l||^2 tr(J_l^T J_l) / p     (invariant)
       d_tilde = sum_l ||delta_l||^2 / ||w_l||^2       (invariant)
   alongside the raw s_bar_J, ||delta||^2 and the per-layer weight norms.

Usage (server, checkpoints in saved_models/):
    python scripts/analyze_existing_checkpoints.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.registry import build_model, build_dataloaders

SEEDS = [13, 42, 97]
SPARSE_RUNS = [
    "ResNet18_CIFAR10_s0.995_shortrecovery",
    "ResNet18_CIFAR10_s0.999_shortrecovery",
    "ResNet18_CIFAR10_s0.9995_shortrecovery",
    "ResNet18_CIFAR10_s0.999_shortrecovery_isocompute_sgd",
    "ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd",
    "VGG16_CIFAR10_s0.999_shortrecovery",
    "VGG16_CIFAR10_s0.9995_shortrecovery",
]
DENSE = [("configs/dense/ResNet18_CIFAR10.json", "ResNet18_cifar10"),
         ("configs/dense/VGG16_CIFAR10.json", "vgg16_bn_cifar10")]
SPARSITIES = [0.5, 0.7, 0.9]
SUBSET_BATCHES = 10  # what the training logs used (eval_batches)
N_HUTCHINSON = 20
JAC_BATCH = 512


def prunable(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def load_any(model_name, model_params, path, device):
    """Load a checkpoint saved either with pruning reparametrization
    (weight_orig/weight_mask) or as plain weights."""
    model = build_model(model_name, model_params).to(device)
    state = torch.load(path, map_location=device)
    if any(k.endswith("weight_orig") for k in state):
        for _, m in prunable(model):
            prune.identity(m, "weight")
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def accuracy(model, loader, device):
    full_c = full_n = sub_c = sub_n = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        c = model(x).argmax(1).eq(y).sum().item()
        full_c += c
        full_n += y.numel()
        if i < SUBSET_BATCHES:
            sub_c += c
            sub_n += y.numel()
    return full_c / full_n, sub_c / sub_n


def per_layer_traces(model, data, n_probe=N_HUTCHINSON):
    """Hutchinson estimate of tr(J_l^T J_l) for every prunable layer, via
    output-space Rademacher probes (one ordinary backward per probe)."""
    model.eval()
    layers = prunable(model)
    gen = torch.Generator(device=data.device).manual_seed(0)
    tr = [0.0] * len(layers)
    for _ in range(n_probe):
        model.zero_grad()
        out = model(data)
        u = torch.randint(0, 2, out.shape, device=out.device, generator=gen).to(out.dtype) * 2 - 1
        (u * out).sum().backward()
        for i, (_, m) in enumerate(layers):
            tr[i] += m.weight.grad.pow(2).sum().item()
    model.zero_grad()
    return [t / n_probe for t in tr]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"full_test_accuracy": {}, "dense_scale_invariant": {}}

    # ---- 1. full-test-set accuracy of final sparse models ----------------
    for run in SPARSE_RUNS:
        cfg = json.load(open(f"configs/sparse/{run}.json"))
        name, params = cfg["model"]["name"], cfg["model"]["parameters"]
        _, test_loader = build_dataloaders(cfg["dataset"]["name"], cfg["dataset"]["batch_size"])
        tag = "ResNet18_cifar10" if name == "ResNet18" else "vgg16_bn_cifar10"
        out["full_test_accuracy"][run] = {}
        for seed in SEEDS:
            for sam in ["True", "False"]:
                path = f"saved_models/sparse/{run}/seed_{seed}/{tag}_sam_{sam}.pth"
                if not os.path.exists(path):
                    continue
                model = load_any(name, params, path, device)
                full, sub = accuracy(model, test_loader, device)
                out["full_test_accuracy"][run].setdefault(sam, {})[str(seed)] = {"full": full, "subset1280": sub}
                print(f"{run} seed={seed} SAM={sam}: full={full:.4f} subset={sub:.4f}")
                del model

    # ---- 2. scale-invariant factors on dense checkpoints -----------------
    for cfg_path, tag in DENSE:
        cfg = json.load(open(cfg_path))
        name, params = cfg["model"]["name"], cfg["model"]["parameters"]
        _, test_loader = build_dataloaders(cfg["dataset"]["name"], JAC_BATCH)
        x, _ = next(iter(test_loader))
        data = x.to(device)
        out["dense_scale_invariant"][tag] = {}
        for seed in SEEDS:
            res = {}
            for sam in ["True", "False"]:
                path = f"saved_models/dense/{tag}/seed_{seed}/{tag}_sam_{sam}.pth"
                model = load_any(name, params, path, device)
                layers = prunable(model)
                w2 = [m.weight.detach().pow(2).sum().item() for _, m in layers]
                numel = [m.weight.numel() for _, m in layers]
                p = sum(numel)
                tr = per_layer_traces(model, data)
                entry = {
                    "layer_names": [n for n, _ in layers],
                    "layer_w2": w2,
                    "layer_trace": tr,
                    "s_bar_J": sum(tr) / p,
                    "s_tilde": sum(a * b for a, b in zip(w2, tr)) / p,
                    "theta2": sum(w2),
                    "per_sparsity": {},
                }
                flat = torch.cat([m.weight.detach().abs().flatten() for _, m in layers])
                for s in SPARSITIES:
                    k = int(s * flat.numel())
                    thr = flat.kthvalue(k).values.item()
                    d2 = [m.weight.detach()[m.weight.detach().abs() <= thr].pow(2).sum().item() for _, m in layers]
                    entry["per_sparsity"][str(s)] = {
                        "delta2": sum(d2),
                        "delta2_over_theta2": sum(d2) / sum(w2),
                        "d_tilde": sum(a / b for a, b in zip(d2, w2)),
                    }
                res[sam] = entry
                print(f"{tag} seed={seed} SAM={sam}: s_bar={entry['s_bar_J']:.4f} s_tilde={entry['s_tilde']:.4f} theta2={entry['theta2']:.1f}")
                del model
            out["dense_scale_invariant"][tag][str(seed)] = res

    os.makedirs("results", exist_ok=True)
    with open("results/existing_checkpoints_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved results/existing_checkpoints_analysis.json")


if __name__ == "__main__":
    main()

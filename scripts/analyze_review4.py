"""Cheap analyses requested by the fourth review round, all from existing
checkpoints (no training):

  1. generous: full-test accuracy of the generous-schedule IMP runs
     (5 cuts at epochs 15-55), which the paper so far reported on the
     1,280-image subset only.
  2. effective: effective sparsity (weights on at least one input-output
     path) of the final IMP models, tight / compute-matched / generous.
  3. jvp: first-order ||J delta||^2 for one-shot pruning of the dense CNNs,
     so that kappa can be split into linear misalignment and nonlinearity
     (Table 13 uses the measured ||Delta f||^2).

Usage (server):
    python scripts/analyze_review4.py generous effective jvp
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.func import functional_call, jvp

from src.registry import build_model, build_dataloaders

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [13, 42, 97]
ARCH = {"ResNet18": ("ResNet18", "ResNet18_cifar10"),
        "VGG16": ("vgg16_bn", "vgg16_bn_cifar10")}


def prunable(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def load_sparse(arch, path):
    """Final IMP checkpoints keep PyTorch's weight_orig/weight_mask pairs."""
    name, _ = ARCH[arch]
    m = build_model(name, {"num_classes": 10}).to(DEV)
    sd = torch.load(path, map_location=DEV)
    if any(k.endswith("weight_orig") for k in sd):
        for _, mod in prunable(m):
            prune.identity(mod, "weight")
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def full_test_acc(model, loader):
    c = n = 0
    for x, y in loader:
        c += model(x.to(DEV)).argmax(1).eq(y.to(DEV)).sum().item()
        n += y.numel()
    return c / n


def masks_of(model):
    out = {}
    for n, mod in prunable(model):
        # module.weight is only refreshed by a forward pass after loading,
        # so read the mask (and the masked weights) directly.
        w = mod.weight_orig * mod.weight_mask if hasattr(mod, "weight_mask") else mod.weight
        out[n] = (w.detach() != 0).double()
    return out


def effective_masks(arch, masks):
    """A weight is effective if it lies on a path from the input to the output.
    Run the network with |weights| = mask, biases 0, BatchNorm as identity and
    an all-ones input: every activation is then non-negative, so the gradient
    of the summed output w.r.t. a weight is positive exactly when its input
    channel is reachable from the input and its output channel reaches the
    output (ReLU and max-pooling only see non-negative values)."""
    name, _ = ARCH[arch]
    m = build_model(name, {"num_classes": 10}).double().to(DEV)
    m.eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm1d):
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
                mod.weight.fill_(1.0); mod.bias.zero_(); mod.eps = 0.0
            if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.bias is not None:
                mod.bias.zero_()
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
        for n, mod in prunable(m):
            mod.weight.copy_(masks[n])
    x = torch.ones(1, 3, 32, 32, dtype=torch.double, device=DEV)
    m.zero_grad()
    m(x).sum().backward()
    eff = {}
    for n, mod in prunable(m):
        eff[n] = ((mod.weight.grad != 0).double() * masks[n])
    return eff


def run_generous(test_loader):
    res = {}
    for arch, sparsities in [("ResNet18", [0.95, 0.995, 0.999, 0.9995]),
                             ("VGG16", [0.7, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995])]:
        _, tag = ARCH[arch]
        for s in sparsities:
            for seed in SEEDS:
                for sam in ["True", "False"]:
                    path = f"saved_models/sparse/{tag}_prune_ratio_{s}/seed_{seed}/{tag}_sam_{sam}.pth"
                    if not os.path.exists(path):
                        continue
                    m = load_sparse(arch, path)
                    nz = sum(int(v.sum().item()) for v in masks_of(m).values())
                    tot = sum(mod.weight.numel() for _, mod in prunable(m))
                    acc = full_test_acc(m, test_loader)
                    key = f"{arch}/{s}/seed_{seed}/SAM_{sam}"
                    res[key] = {"acc": acc, "density": nz / tot}
                    print(key, f"acc={acc:.4f} density={nz / tot:.6f}", flush=True)
    json.dump(res, open("results/generous_fulltest.json", "w"), indent=1)


def run_effective():
    runs = []
    for arch, folder_fmt in [("ResNet18", "ResNet18_CIFAR10_s{s}_shortrecovery"),
                             ("VGG16", "VGG16_CIFAR10_s{s}_shortrecovery")]:
        _, tag = ARCH[arch]
        for s in ["0.999", "0.9995"]:
            for seed in SEEDS:
                for sam in ["True", "False"]:
                    runs.append((arch, f"tight/{s}", f"saved_models/sparse/{folder_fmt.format(s=s)}/seed_{seed}/{tag}_sam_{sam}.pth", seed, sam))
                runs.append((arch, f"isocompute/{s}", f"saved_models/sparse/{folder_fmt.format(s=s)}_isocompute_sgd/seed_{seed}/{tag}_sam_False.pth", seed, "False"))
            for seed in SEEDS:
                for sam in ["True", "False"]:
                    runs.append((arch, f"generous/{s}", f"saved_models/sparse/{tag}_prune_ratio_{s}/seed_{seed}/{tag}_sam_{sam}.pth", seed, sam))
    res = {}
    for arch, kind, path, seed, sam in runs:
        if not os.path.exists(path):
            continue
        m = load_sparse(arch, path)
        masks = masks_of(m)
        eff = effective_masks(arch, masks)
        nom = {n: int(v.sum().item()) for n, v in masks.items()}
        effc = {n: int(v.sum().item()) for n, v in eff.items()}
        numel = {n: v.numel() for n, v in masks.items()}
        key = f"{arch}/{kind}/seed_{seed}/SAM_{sam}"
        res[key] = {"nominal": sum(nom.values()), "effective": sum(effc.values()),
                    "total": sum(numel.values()),
                    "per_layer_nominal": nom, "per_layer_effective": effc, "numel": numel}
        print(key, "nominal", sum(nom.values()), "effective", sum(effc.values()), flush=True)
    json.dump(res, open("results/effective_sparsity.json", "w"), indent=1)


def run_jvp(test_loader):
    xs = []
    for x, _ in test_loader:
        xs.append(x)
        if sum(t.shape[0] for t in xs) >= 512:
            break
    data = torch.cat(xs)[:512].to(DEV)
    res = {}
    for arch in ["ResNet18", "VGG16"]:
        name, tag = ARCH[arch]
        for seed in SEEDS:
            for sam in ["True", "False"]:
                path = f"saved_models/dense/{tag}/seed_{seed}/{tag}_sam_{sam}.pth"
                for s in [0.5, 0.7, 0.9]:
                    m = build_model(name, {"num_classes": 10}).to(DEV)
                    m.load_state_dict(torch.load(path, map_location=DEV))
                    m.eval()
                    pairs = prunable(m)
                    prune.global_unstructured([(mod, "weight") for _, mod in pairs],
                                              pruning_method=prune.L1Unstructured, amount=s)
                    # delta = theta_pruned - theta_dense on the prunable weights
                    tangents, primals = {}, {}
                    for n, mod in pairs:
                        w = mod.weight_orig.detach()
                        primals[f"{n}.weight"] = w
                        tangents[f"{n}.weight"] = -(w * (1 - mod.weight_mask))
                    for _, mod in pairs:
                        prune.remove(mod, "weight")
                    # restore dense weights for the linearization point
                    with torch.no_grad():
                        for n, mod in pairs:
                            mod.weight.copy_(primals[f"{n}.weight"])

                    def f(params):
                        return functional_call(m, params, (data,))

                    with torch.no_grad():
                        _, jd = jvp(f, (primals,), (tangents,))
                    lin = jd.pow(2).sum().item()
                    d_sq = sum(t.pow(2).sum().item() for t in tangents.values())
                    key = f"{arch}/seed_{seed}/SAM_{sam}/{s}"
                    res[key] = {"Jdelta_sq": lin, "delta_sq": d_sq}
                    print(key, f"||J delta||^2={lin:.4f} ||delta||^2={d_sq:.4f}", flush=True)
                    del m
    json.dump(res, open("results/jvp_first_order_cnn.json", "w"), indent=1)


if __name__ == "__main__":
    _, test_loader = build_dataloaders("cifar10", 512)
    tasks = sys.argv[1:] or ["generous", "effective", "jvp"]
    if "generous" in tasks:
        run_generous(test_loader)
    if "effective" in tasks:
        run_effective()
    if "jvp" in tasks:
        run_jvp(test_loader)

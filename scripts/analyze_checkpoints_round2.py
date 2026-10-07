"""Second round of analyses on existing checkpoints (no training).

1. Output-scale-normalized sensitivity (dense checkpoints). s_bar_J is the
   mean squared Jacobian of the logits, so a model with larger logits has a
   larger s_bar_J without being more sensitive in any way that matters for
   predictions. We report, on the same fixed batch of 512 test images:
     - s_bar_J of the raw logits,
     - s_bar_J of the centered logits f - mean_c f (softmax ignores shifts),
     - the centered-logit variance v = mean ||f_c||^2 / C,
     - s_norm = s_bar_J(centered) / v, invariant to rescaling the logits,
     - the mean top-1/top-2 logit margin.
2. Restricted gradient norm before and after one-shot pruning, averaged over
   the same 20 training batches (eval mode, as in post_pruning_metrics), so
   the dense model's own non-stationarity can be separated from the cut.
3. Layer collapse in the final iterative-pruning models: number of
   Conv/Linear layers left with no active weight, and the smallest per-layer
   density.
4. Parameter counts of the prunable weights.

Usage (server):
    python scripts/analyze_checkpoints_round2.py
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
DENSE = [("configs/dense/ResNet18_CIFAR10.json", "ResNet18_cifar10"),
         ("configs/dense/VGG16_CIFAR10.json", "vgg16_bn_cifar10")]
SPARSE_RUNS = [
    "ResNet18_CIFAR10_s0.995_shortrecovery",
    "ResNet18_CIFAR10_s0.999_shortrecovery",
    "ResNet18_CIFAR10_s0.9995_shortrecovery",
    "VGG16_CIFAR10_s0.999_shortrecovery",
    "VGG16_CIFAR10_s0.9995_shortrecovery",
]
SPARSITIES = [0.5, 0.7, 0.9]
N_PROBE = 20
GRAD_BATCHES = 20


def prunable(model):
    return [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def load_any(name, params, path, device):
    model = build_model(name, params).to(device)
    state = torch.load(path, map_location=device)
    if any(k.endswith("weight_orig") for k in state):
        for m in prunable(model):
            prune.identity(m, "weight")
    model.load_state_dict(state)
    return model.eval()


def sensitivity(model, data):
    model.eval()
    weights = [m.weight for m in prunable(model)]
    p = sum(w.numel() for w in weights)
    gen = torch.Generator(device=data.device).manual_seed(0)
    raw = cen = 0.0
    for _ in range(N_PROBE):
        for centered in (False, True):
            model.zero_grad()
            out = model(data)
            if centered:
                out = out - out.mean(dim=1, keepdim=True)
            u = torch.randint(0, 2, out.shape, device=out.device, generator=gen).to(out.dtype) * 2 - 1
            (u * out).sum().backward()
            val = sum(w.grad.pow(2).sum().item() for w in weights)
            if centered:
                cen += val
            else:
                raw += val
    model.zero_grad()
    with torch.no_grad():
        out = model(data)
        fc = out - out.mean(dim=1, keepdim=True)
        var = fc.pow(2).mean().item()
        top2 = out.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
    s_raw, s_cen = raw / N_PROBE / p, cen / N_PROBE / p
    return {"s_bar_J": s_raw, "s_bar_J_centered": s_cen, "logit_var": var,
            "s_norm": s_cen / var, "margin": margin, "p": p}


def grad_norm(model, batches, device, criterion):
    """Mean restricted gradient norm over fixed batches (eval mode)."""
    model.eval()
    vals = []
    for x, y in batches:
        model.zero_grad()
        criterion(model(x.to(device)), y.to(device)).backward()
        sq = 0.0
        for mod in model.modules():
            for pname, prm in mod.named_parameters(recurse=False):
                if prm.grad is None:
                    continue
                mask = getattr(mod, pname.replace("_orig", "") + "_mask", None) if pname.endswith("_orig") else None
                g = prm.grad * mask if mask is not None else prm.grad
                sq += g.pow(2).sum().item()
        vals.append(sq ** 0.5)
    model.zero_grad()
    return sum(vals) / len(vals)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    out = {"dense": {}, "collapse": {}, "params": {}}

    for cfg_path, tag in DENSE:
        cfg = json.load(open(cfg_path))
        name, params = cfg["model"]["name"], cfg["model"]["parameters"]
        train_loader, test_loader = build_dataloaders(cfg["dataset"]["name"], 512)
        data = next(iter(test_loader))[0].to(device)
        tl, _ = build_dataloaders(cfg["dataset"]["name"], 128)
        torch.manual_seed(0)
        batches = []
        for i, b in enumerate(tl):
            if i >= GRAD_BATCHES:
                break
            batches.append(b)
        out["dense"][tag] = {}
        for seed in SEEDS:
            res = {}
            for sam in ["True", "False"]:
                path = f"saved_models/dense/{tag}/seed_{seed}/{tag}_sam_{sam}.pth"
                model = load_any(name, params, path, device)
                entry = sensitivity(model, data)
                entry["grad_pre"] = grad_norm(model, batches, device, criterion)
                entry["grad_post"] = {}
                for s in SPARSITIES:
                    pm = load_any(name, params, path, device)
                    prune.global_unstructured([(m, "weight") for m in prunable(pm)],
                                              pruning_method=prune.L1Unstructured, amount=s)
                    entry["grad_post"][str(s)] = grad_norm(pm, batches, device, criterion)
                    del pm
                res[sam] = entry
                out["params"][tag] = entry["p"]
                print(f"{tag} seed={seed} SAM={sam}: s_bar={entry['s_bar_J']:.4f} s_cen={entry['s_bar_J_centered']:.4f} "
                      f"var={entry['logit_var']:.2f} s_norm={entry['s_norm']:.5f} margin={entry['margin']:.2f} "
                      f"grad_pre={entry['grad_pre']:.4f} grad_post={entry['grad_post']}", flush=True)
                del model
            out["dense"][tag][str(seed)] = res

    for run in SPARSE_RUNS:
        cfg = json.load(open(f"configs/sparse/{run}.json"))
        name, params = cfg["model"]["name"], cfg["model"]["parameters"]
        tag = "ResNet18_cifar10" if name == "ResNet18" else "vgg16_bn_cifar10"
        out["collapse"][run] = {}
        for seed in SEEDS:
            for sam in ["True", "False"]:
                path = f"saved_models/sparse/{run}/seed_{seed}/{tag}_sam_{sam}.pth"
                if not os.path.exists(path):
                    continue
                state = torch.load(path, map_location="cpu")
                dens = []
                for k, v in state.items():
                    if k.endswith("weight_mask"):
                        dens.append(v.float().mean().item())
                out["collapse"][run][f"{seed}|{sam}"] = {"n_layers": len(dens),
                                                       "n_empty": sum(d == 0 for d in dens),
                                                       "min_density": min(dens)}
                print(f"{run} seed={seed} SAM={sam}: layers={len(dens)} empty={sum(d == 0 for d in dens)} "
                      f"min_density={min(dens):.2e}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/checkpoints_round2.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved results/checkpoints_round2.json")


if __name__ == "__main__":
    main()

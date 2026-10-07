"""Linear mode connectivity across iterative-pruning rounds, SAM vs SGD.

Uses the per-5-epoch checkpoints of the fully checkpointed seed-13 runs.
Pruning happens at the START of epochs 15, 30, ..., 165, and a checkpoint at
epoch e holds the weights at the END of epoch e (pruned weights stored as
exact zeros). Two kinds of pairs per round:

  successive: theta(c+10) -> theta(c+25)
      late-window solutions of consecutive rounds (Paul et al., ICLR 2023:
      successive IMP iterates are linearly connected iff they are matching).
  recovery:   m_{c'} * theta(c'-5) -> theta(c'+10)
      the pruned point (next mask applied to the late-window weights; the
      last 4 epochs before the cut are not checkpointed) and the retrained
      solution 10 epochs after the cut. Both endpoints share mask m_{c'},
      so the path stays inside the pruned subspace U_m.

At every interpolation point BatchNorm statistics are recomputed on
training data (standard for BN networks, Frankle et al. 2020), then test
error and loss are measured on the full 10,000-image test set. The barrier
is max_alpha [metric(alpha) - linear interpolation of the endpoints].

Usage (server):
    python scripts/analyze_linear_connectivity.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.cifar10 import load as load_cifar10
from src.registry import build_model

RUNS = [
    "ResNet18_CIFAR10_s0.995_shortrecovery",
    "ResNet18_CIFAR10_s0.999_shortrecovery",
    "ResNet18_CIFAR10_s0.9995_shortrecovery",
    "VGG16_CIFAR10_s0.999_shortrecovery",
    "VGG16_CIFAR10_s0.9995_shortrecovery",
]
SEED = 13
CUTS = list(range(15, 166, 15))
ALPHAS = [i / 8 for i in range(9)]
BN_BATCHES = 20
BATCH = 256
OUT = "results/linear_connectivity_seed13.json"


def ckpt(run, sam, epoch):
    return f"saved_models/sparse/{run}/seed_{SEED}/checkpoint/sam_{sam}_epoch_{epoch}.pth"


def mask_of(state):
    return {k: (v != 0).float() for k, v in state.items() if k.endswith("weight") and v.dim() > 1}


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the recalibration batches
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= BN_BATCHES:
            break
        model(x.to(device))
    model.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    wrong = loss = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        wrong += (out.argmax(1) != y).sum().item()
        n += y.numel()
    return wrong / n, loss / n


def interpolate(model, a, b, train_loader, test_loader, device):
    curve = []
    for alpha in ALPHAS:
        state = {k: (1 - alpha) * a[k] + alpha * b[k] if a[k].is_floating_point() else a[k] for k in a}
        model.load_state_dict(state)
        recalibrate_bn(model, train_loader, device)
        err, loss = evaluate(model, test_loader, device)
        curve.append({"alpha": alpha, "err": err, "loss": loss})
    return curve


def barrier(curve, key):
    lo, hi = curve[0][key], curve[-1][key]
    return max(c[key] - ((1 - c["alpha"]) * lo + c["alpha"] * hi) for c in curve)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, testset = load_cifar10(randomize=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2,
                                               generator=torch.Generator().manual_seed(0))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for run in RUNS:
        cfg = json.load(open(f"configs/sparse/{run}.json"))
        model = build_model(cfg["model"]["name"], cfg["model"]["parameters"]).to(device)
        for sam in ["True", "False"]:
            key = f"{run}|{sam}"
            if key in results:
                continue
            load = lambda e: torch.load(ckpt(run, sam, e), map_location=device)
            entry = {"successive": [], "recovery": []}
            for i, c in enumerate(CUTS):
                # recovery pair around cut c (needs the checkpoint 5 epochs before it)
                pre = load(c - 5)
                post_state = load(c)  # end of the cut epoch: zero pattern = mask m_c
                m = mask_of(post_state)
                pruned = {k: v * m[k] if k in m else v for k, v in pre.items()}
                curve = interpolate(model, pruned, load(c + 10), train_loader, test_loader, device)
                entry["recovery"].append({"cut": c, "curve": curve,
                                          "err_barrier": barrier(curve, "err"),
                                          "loss_barrier": barrier(curve, "loss")})
                # successive late-window solutions of rounds c and c+15
                if i + 1 < len(CUTS):
                    curve = interpolate(model, load(c + 10), load(c + 25), train_loader, test_loader, device)
                    entry["successive"].append({"from": c + 10, "to": c + 25, "curve": curve,
                                                "err_barrier": barrier(curve, "err"),
                                                "loss_barrier": barrier(curve, "loss")})
                r = entry["recovery"][-1]
                print(f"{key} cut={c}: recovery err-barrier={r['err_barrier']:.4f}"
                      + (f"  successive err-barrier={entry['successive'][-1]['err_barrier']:.4f}" if i + 1 < len(CUTS) else ""),
                      flush=True)
            results[key] = entry
            os.makedirs("results", exist_ok=True)
            with open(OUT, "w") as f:
                json.dump(results, f)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

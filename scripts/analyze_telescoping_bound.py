"""Test the telescoping extension of Proposition 3.1' against real checkpoints.

Loads the saved seed-13 per-round checkpoints from the tight-recovery sweep
(ResNet18, s=0.995 and s=0.999 -- the two configs with the biggest measured
SAM-vs-SGD final-accuracy gap) and, at every saved epoch, for both SAM and
SGD, measures:
  - the restricted gradient norm ||P_m^T grad L(theta)|| (masked_grad_norm)
  - lambda1(H_m), the top eigenvalue of the restricted Hessian
  - for SAM only: the predicted beta1 = eta*rho*lambda1^2 / (2 - eta*lambda1)
    (Bartlett et al.'s own oscillation term, Lemma 5), to compare against
    the measured residual directly

This is the quantitative test of the telescoping-bound conjecture from
archive/FUTURE_WORK_A1_RELAXATION.md and rebuttal/YZBM.md Q1 / KPwu.md W2:
does SAM's measured residual track beta1 while SGD's grows uncontrolled
round over round, especially once the 15-epoch recovery window stops being
enough to reconverge?

Saved checkpoints don't retain the pruning reparametrization (train_prune_loop
calls prune.remove() before saving), so this re-derives each checkpoint's
mask from which weight entries are exactly zero and reattaches it with
prune.custom_from_mask -- after which the existing masked_grad_norm and
hessian.pruned_eigenvalues() machinery works unmodified.

Usage (run on the server, in the venv that has the checkpoints):
    python scripts/analyze_telescoping_bound.py \
        --config configs/sparse/ResNet18_CIFAR10_s0.995_shortrecovery.json \
        --seed 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Running this as `python scripts/analyze_telescoping_bound.py` only puts
# scripts/ on sys.path, not the repo root -- add the root explicitly so
# `from src...` resolves regardless of where this is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.registry import build_model, build_dataloaders, build_criterion
from src.pyhessian.hessian import hessian
from src.eval.eval import masked_grad_norm

GRAD_BATCH_SIZE = 2048
HESSIAN_MAX_ITER = 30
# Round boundaries are expensive (Hessian power iteration); gradient norm is
# cheap (one backward pass) so we compute it at every saved checkpoint.
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]


def get_big_batch(loader, device, n):
    xs, ys = [], []
    collected = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        collected += x.shape[0]
        if collected >= n:
            break
    return torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)


def reattach_masks(model: nn.Module) -> int:
    """Derive each Linear/Conv2d weight's mask from its zero pattern and
    reattach it with prune.custom_from_mask, so masked_grad_norm and
    hessian.pruned_eigenvalues() see the same weight_mask buffers the live
    training run had. Returns the number of active (nonzero) weight entries.
    """
    active = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            mask = (module.weight.detach() != 0).float()
            active += int(mask.sum().item())
            prune.custom_from_mask(module, "weight", mask=mask)
    return active


def load_checkpoint(model_name, model_params, ckpt_path, device):
    model = build_model(model_name, model_params).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    active = reattach_masks(model)
    return model, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = json.load(open(args.config))
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    eta = config["training"]["learning_rate"]
    rho = config["training"]["rho"]

    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = os.path.join(
        "saved_models", "sparse", config_tag, f"seed_{args.seed}", "checkpoint"
    )
    out_path = os.path.join("results", f"telescoping_{config_tag}_seed{args.seed}.json")
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_criterion(config["training"]["loss_function"])
    train_loader, _ = build_dataloaders(dataset_name, batch_size)
    big_data, big_target = get_big_batch(train_loader, device, GRAD_BATCH_SIZE)

    available = sorted(
        int(f.split("_epoch_")[1].split(".pth")[0])
        for f in os.listdir(ckpt_dir)
        if f.startswith("sam_True_epoch_")
    )
    print(f"{config_tag}: {len(available)} checkpoints available: {available}")

    results = {"True": [], "False": []}
    for use_sam_str in ["True", "False"]:
        for epoch in available:
            ckpt_path = os.path.join(ckpt_dir, f"sam_{use_sam_str}_epoch_{epoch}.pth")
            model, active = load_checkpoint(model_name, model_params, ckpt_path, device)

            model.zero_grad()
            out = model(big_data)
            loss = criterion(out, big_target).mean()
            loss.backward()
            grad_norm = masked_grad_norm(model).item()
            model.zero_grad()

            row = {"epoch": epoch, "active_params": active, "grad_norm": grad_norm}

            if epoch in ROUND_EPOCHS:
                cuda = device.type == "cuda"
                hessian_comp = hessian(model, criterion, data=(big_data, big_target), cuda=cuda)
                eigvals, _ = hessian_comp.pruned_eigenvalues(top_n=1, maxIter=HESSIAN_MAX_ITER)
                lambda1 = eigvals[0]
                eta_lambda1 = eta * lambda1
                beta1_pred = (eta * rho * lambda1 ** 2) / max(2 - eta_lambda1, 1e-6)
                row["lambda1"] = lambda1
                row["beta1_pred"] = beta1_pred
                print(
                    f"  SAM={use_sam_str} epoch={epoch:3d}  active={active:>9,d}  "
                    f"grad_norm={grad_norm:10.4f}  lambda1={lambda1:10.2f}  "
                    f"beta1_pred={beta1_pred:10.4f}"
                )
            else:
                print(f"  SAM={use_sam_str} epoch={epoch:3d}  active={active:>9,d}  grad_norm={grad_norm:10.4f}")

            results[use_sam_str].append(row)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

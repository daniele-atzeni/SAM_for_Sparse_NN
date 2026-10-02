"""Three-factor decomposition (Proposition 3.2) on CNN checkpoints, the
CNN-side validation Table 1 only has for MLPs.

All three raw quantities are computable from existing dense checkpoints
with standard backward passes -- no Jacobian power iteration needed,
since R_s in Prop. 3.2 is the *trace*-based mean sensitivity
s_bar_J = tr(J^T J) / p, not the top singular value:

  - s_bar_J, via Hutchinson trace estimation in OUTPUT space: for random
    Rademacher u in R^{nC} (same shape as the stacked batch logits),
    tr(J^T J) = tr(J J^T) = E_u[||J^T u||^2], and J^T u is just the
    gradient of (u . f(theta)) w.r.t. theta -- one ordinary backward
    pass, no double-backward. Evaluated once at the dense checkpoint
    (constant across sparsity levels, matching Table 1's structure).
  - ||delta||^2, the squared norm of the pruned-away weight values --
    trivial, no forward/backward needed.
  - ||Delta f||^2 = ||f(theta^(m)) - f(theta*)||^2, the squared norm of
    the change in stacked logits after pruning -- two forward passes.

R_delta, Actual, and the implied alignment ratio eta are then computed
from these per sparsity level, same as the MLP table.

Usage (run on the server, in the venv that has the checkpoints):
    python scripts/analyze_three_factor_cnn.py \
        --dense-config configs/dense/ResNet18_CIFAR10.json \
        --dense-tag ResNet18_cifar10 --seed 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.registry import build_model, build_dataloaders

SPARSITIES = [0.5, 0.7, 0.9]
BATCH_SIZE = 512  # nC-dim output space for the Hutchinson trace estimate; kept
                   # modest since we need the full stacked-logit vector in memory.
N_HUTCHINSON = 20


def get_batch(loader, device, n):
    xs, ys = [], []
    collected = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        collected += x.shape[0]
        if collected >= n:
            break
    return torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)


def prunable_weights(model):
    """Conv/Linear weights -- the same parameter set magnitude pruning acts
    on, so s_bar_J and delta live on the same coordinates."""
    return [m.weight for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def s_bar_J(model, data, n_hutchinson=N_HUTCHINSON):
    """Hutchinson estimate of tr(J^T J) / p for the restricted Jacobian,
    via VJPs only (random linear combinations of the stacked logits)."""
    model.eval()  # BN must use running stats, as at deployment
    weights = prunable_weights(model)
    p = sum(w.numel() for w in weights)
    gen = torch.Generator(device=data.device).manual_seed(0)
    total = 0.0
    for _ in range(n_hutchinson):
        model.zero_grad()
        out = model(data)  # [batch, C]
        u = torch.randint(0, 2, out.shape, device=out.device, generator=gen).to(out.dtype) * 2 - 1  # Rademacher
        (u * out).sum().backward()
        sq_norm = sum(w.grad.pow(2).sum().item() for w in weights)
        total += sq_norm
    model.zero_grad()
    return (total / n_hutchinson) / p, p


@torch.no_grad()
def delta_sq_norm(dense_state, pruned_model):
    """||delta||^2 = sum of the squared values of the weights that got
    pruned away (delta has support only on pruned coordinates by
    construction of magnitude pruning)."""
    total = 0.0
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            key = f"{name}.weight"
            if key not in dense_state:
                continue
            mask = getattr(module, "weight_mask", torch.ones_like(dense_state[key]))
            pruned_entries = dense_state[key] * (1 - mask)
            total += pruned_entries.pow(2).sum().item()
    return total


@torch.no_grad()
def delta_f_sq_norm(dense_model, pruned_model, data):
    dense_model.eval()
    pruned_model.eval()
    f_dense = dense_model(data)
    f_pruned = pruned_model(data)
    return (f_pruned - f_dense).pow(2).sum().item()


def prune_copy(model_name, model_params, dense_path, sparsity, device):
    model = build_model(model_name, model_params).to(device)
    state = torch.load(dense_path, map_location=device)
    model.load_state_dict(state)
    params_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity)
    return model, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-config", required=True)
    parser.add_argument("--dense-tag", required=True, help="e.g. ResNet18_cifar10")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = json.load(open(args.dense_config))
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    dataset_name = config["dataset"]["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed, unaugmented batch (test loader: shuffle=False) so every run and
    # every checkpoint is evaluated on exactly the same inputs.
    _, test_loader = build_dataloaders(dataset_name, BATCH_SIZE)
    data, _ = get_batch(test_loader, device, BATCH_SIZE)

    dense_dir = os.path.join("saved_models", "dense", args.dense_tag, f"seed_{args.seed}")
    results = {"s_bar_J": {}, "active_params": {}, "per_sparsity": {}}

    for sam_flag in ["True", "False"]:
        dense_path = os.path.join(dense_dir, f"{args.dense_tag}_sam_{sam_flag}.pth")
        model = build_model(model_name, model_params).to(device)
        state = torch.load(dense_path, map_location=device)
        model.load_state_dict(state)
        sbar, p = s_bar_J(model, data)
        results["s_bar_J"][sam_flag] = sbar
        results["active_params"][sam_flag] = p
        print(f"SAM={sam_flag}: s_bar_J={sbar:.6f}  (p={p})")
        del model

    for s in SPARSITIES:
        results["per_sparsity"][str(s)] = {}
        for sam_flag in ["True", "False"]:
            dense_path = os.path.join(dense_dir, f"{args.dense_tag}_sam_{sam_flag}.pth")
            dense_model = build_model(model_name, model_params).to(device)
            dense_model.load_state_dict(torch.load(dense_path, map_location=device))
            dense_state = {k: v.clone() for k, v in dense_model.state_dict().items()}

            pruned_model, _ = prune_copy(model_name, model_params, dense_path, s, device)

            d_sq = delta_sq_norm(dense_state, pruned_model)
            df_sq = delta_f_sq_norm(dense_model, pruned_model, data)
            results["per_sparsity"][str(s)][sam_flag] = {"delta_sq": d_sq, "delta_f_sq": df_sq}
            print(f"s={s} SAM={sam_flag}: ||delta||^2={d_sq:.4f}  ||Delta f||^2={df_sq:.4f}")
            del dense_model, pruned_model

    out_path = f"results/three_factor_cnn_{args.dense_tag}_seed{args.seed}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

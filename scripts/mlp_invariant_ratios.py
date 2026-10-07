"""Per-layer rescaling-invariant ratios for the MLPs of Table 1.

A ReLU MLP is invariant to w_l -> c w_l, w_{l+1} -> w_{l+1}/c, which changes
the raw ||delta||^2 and tr(J^T J) but not
    d_tilde = sum_l ||delta_l||^2 / ||w_l||^2,
    s_tilde = sum_l ||w_l||^2 tr(J_l^T J_l) / p,
the quantities already reported for the CNNs (Table 12). The original
decomposition runs did not keep the trained models, so this retrains them
with the same configs and seeds (SGD and SAM from the same init) and also
recomputes the raw R_delta as a check against Table 1.

Usage (server):
    python scripts/mlp_invariant_ratios.py MNIST
    python scripts/mlp_invariant_ratios.py fashionMNIST
"""

from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

from src.registry import build_model, build_dataloaders, build_criterion, build_scheduler
from src.train.SAM import SAM
from src.train.training import train_epoch

CONFIGS = {"MNIST": "configs/multiseed/MLP_MNIST_config.json",
           "fashionMNIST": "configs/multiseed/MLP_FashionMNIST_config.json"}
SEEDS = [42, 43, 44, 45, 46]
SPARSITIES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def per_layer_jac_traces(model, x):
    """Exact tr(J_l^T J_l) over the weights of each Linear layer, on inputs x."""
    model.eval()
    layers = linears(model)
    tr = [0.0] * len(layers)
    for i in range(x.shape[0]):
        out = model(x[i:i + 1])
        for c in range(out.shape[1]):
            grads = torch.autograd.grad(out[0, c], [l.weight for l in layers], retain_graph=c < out.shape[1] - 1)
            for j, g in enumerate(grads):
                tr[j] += g.pow(2).sum().item()
    return tr


def pruned_energy(model, s):
    """Per-layer ||delta_l||^2 under global magnitude pruning at sparsity s."""
    m = copy.deepcopy(model)
    layers = linears(m)
    prune.global_unstructured([(l, "weight") for l in layers], pruning_method=prune.L1Unstructured, amount=s)
    return [(l.weight_orig * (1 - l.weight_mask)).pow(2).sum().item() for l in layers]


def main(dataset):
    config = json.load(open(CONFIGS[dataset]))
    tr_cfg = config["training"]
    lr = tr_cfg["learning_rate"]
    train_loader, _ = build_dataloaders(config["dataset"]["name"], config["dataset"]["batch_size"])
    criterion = build_criterion(tr_cfg["loss_function"])
    n_jac = config.get("max_samples_jacobian", 100)
    out_path = f"results/mlp_invariant_ratios_{dataset}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for seed in SEEDS:
        if f"seed_{seed}" in res:
            continue
        torch.manual_seed(seed)
        model = build_model(config["model"]["name"], config["model"]["parameters"])
        init = copy.deepcopy(model.state_dict())
        trained = {}
        for use_sam in [False, True]:
            model.load_state_dict(init)
            model = model.to(DEV)
            sgd = optim.SGD(model.parameters(), lr=lr, momentum=tr_cfg["momentum"], weight_decay=tr_cfg["weight_decay"])
            sam = SAM(model.parameters(), optim.SGD, rho=tr_cfg["rho"], adaptive=False, lr=lr,
                      momentum=tr_cfg["momentum"], weight_decay=tr_cfg["weight_decay"])
            sched = build_scheduler(config, lr)
            for epoch in range(1, tr_cfg["epochs"] + 1):
                train_epoch(model, DEV, train_loader, sam if use_sam else sgd, epoch, criterion, log_every=10 ** 9)
                sched(sgd, epoch)
                sched(sam, epoch)
            trained["SAM" if use_sam else "SGD"] = copy.deepcopy(model).eval()
            os.makedirs("saved_models/mlp_invariant", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/mlp_invariant/{dataset}_seed{seed}_{'SAM' if use_sam else 'SGD'}.pth")
            print(f"{dataset} seed {seed} {'SAM' if use_sam else 'SGD'} trained", flush=True)

        # Same fixed Jacobian inputs for both models.
        g = torch.Generator().manual_seed(seed)
        ds = train_loader.dataset
        idx = torch.randperm(len(ds), generator=g)[:n_jac]
        x = torch.stack([ds[i][0] for i in idx]).to(DEV)

        row = {}
        for tag, m in trained.items():
            w2 = [l.weight.detach().pow(2).sum().item() for l in linears(m)]
            p = sum(l.weight.numel() for l in linears(m))
            trs = per_layer_jac_traces(m, x)
            row[tag] = {"w2": w2, "jac_tr": trs, "p": p,
                        "s_raw": sum(trs) / p,
                        "s_tilde": sum(a * b for a, b in zip(w2, trs)) / p,
                        "delta": {str(s): pruned_energy(m, s) for s in SPARSITIES}}
        a, b = row["SAM"], row["SGD"]
        row["R_s_weights"] = a["s_raw"] / b["s_raw"]
        row["R_s_tilde"] = a["s_tilde"] / b["s_tilde"]
        row["R_delta"] = {s: sum(a["delta"][s]) / sum(b["delta"][s]) for s in a["delta"]}
        row["R_delta_tilde"] = {s: sum(d / w for d, w in zip(a["delta"][s], a["w2"])) /
                                   sum(d / w for d, w in zip(b["delta"][s], b["w2"])) for s in a["delta"]}
        print(f"seed {seed}: R_s(weights)={row['R_s_weights']:.3f} R_s_tilde={row['R_s_tilde']:.3f}", flush=True)
        print("  R_delta      ", {k: round(v, 3) for k, v in row["R_delta"].items()}, flush=True)
        print("  R_delta_tilde", {k: round(v, 3) for k, v in row["R_delta_tilde"].items()}, flush=True)
        res[f"seed_{seed}"] = row
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)

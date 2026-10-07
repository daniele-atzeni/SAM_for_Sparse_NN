"""Damage per cut and recovery, SAM vs SGD, from the linear-connectivity
analysis (results/linear_connectivity_seed13.json, seed 13).

For each pruning round: test error of the pruned point (BN recalibrated)
and of the solution 10 epochs after the cut, i.e. the two endpoints of the
recovery interpolation path.

Usage:
    python scripts/make_cut_damage_figure.py
"""

import json
import os

import matplotlib.pyplot as plt

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
RUNS = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet-18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet-18, s = 0.999"),
    ("ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet-18, s = 0.9995"),
    ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG-16, s = 0.999"),
    ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG-16, s = 0.9995"),
]

data = json.load(open(os.path.join("results", "linear_connectivity_seed13.json")))
fig, grid = plt.subplots(2, 5, figsize=(15, 6.0), sharey="row", sharex=True)
axes = grid[0]
for col, (ax, (run, title)) in enumerate(zip(axes, RUNS)):
    for sam, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        rec = data[f"{run}|{sam}"]["recovery"]
        cuts = [r["cut"] for r in rec]
        pruned = [100 * r["curve"][0]["err"] for r in rec]
        retrained = [100 * r["curve"][-1]["err"] for r in rec]
        ax.plot(cuts, pruned, color=color, linestyle="--", marker="o", markersize=3, linewidth=1.3,
                label=f"{label}, pruned point")
        ax.plot(cuts, retrained, color=color, marker="o", markersize=3, linewidth=1.6,
                label=f"{label}, 10 epochs later")
        # increase caused by each cut: pruned point minus the pre-cut model,
        # i.e. the retrained endpoint of the previous recovery path
        inc = [p - q for p, q in zip(pruned[1:], retrained[:-1])]
        grid[1, col].plot(cuts[1:], inc, color=color, marker="o", markersize=3, linewidth=1.6, label=label)
    ax.set_title(title, fontsize=11)
    grid[1, col].set_xlabel("pruning epoch")
    for a in (ax, grid[1, col]):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
axes[0].set_ylabel("test error (%)")
grid[1, 0].set_ylabel("error increase from cut (pts)")
grid[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join("figures", f"cut_damage.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")

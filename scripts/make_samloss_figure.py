"""L_SAM (loss after SAM's worst-case perturbation) over training -- the
direct analog of the old paper's Figures 7-12, rebuilt on the verified,
bug-fixed wall x short-recovery data. Logged for both optimizers as a
landscape diagnostic (SGD models get probed with a SAM-style perturbation
at eval time too, even though they weren't trained with one), every 5
epochs, same sampling as the Hessian trace.

Usage:
    python scripts/make_samloss_figure.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
SEEDS = ["13", "42", "97"]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    DATA = json.load(f)

CONFIGS = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18, s = 0.999"),
    ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG16, s = 0.999"),
    ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG16, s = 0.9995"),
]


def mean_std(config, sam_flag, tag):
    per_seed, epochs_ref = [], None
    for seed in SEEDS:
        rows = sorted(DATA[config][seed][sam_flag][tag], key=lambda r: r["step"])
        epochs = [r["step"] for r in rows]
        values = [r["value"] for r in rows]
        if epochs_ref is None:
            epochs_ref = epochs
        per_seed.append(values)
    arr = np.array(per_seed)
    return np.array(epochs_ref), arr.mean(axis=0), arr.std(axis=0)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

for ax, (config, title) in zip(axes.flat, CONFIGS):
    for sam_flag, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        epochs, mean, std = mean_std(config, sam_flag, "SAM Loss/test")
        ax.plot(epochs, mean, marker="o", markersize=3, linewidth=1.8, color=color, label=label, zorder=3)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, zorder=2, linewidth=0)

    for re in ROUND_EPOCHS:
        ax.axvline(re, color="#8C8C8C", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1, alpha=0.75)

    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold")
    style_axes(ax)

for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$L_{\mathrm{SAM}}$ (log scale)")

axes[0, 0].legend(frameon=False, loc="upper left", fontsize=10)
fig.suptitle(
    "Sharpness diagnostic ($L_{\\mathrm{SAM}}$) through training, tight-recovery schedule\n"
    "(mean ± std across 3 seeds; dashed lines mark the 11 pruning rounds)",
    fontsize=12, y=0.985,
)
fig.subplots_adjust(top=0.86, hspace=0.32)

for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"samloss_curves.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
plt.close(fig)

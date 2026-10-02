"""Same data as make_samloss_figure.py, but split into one figure per
architecture (ResNet-18, VGG-16) to match how the paper cites them
separately in the appendix, instead of one combined 2x2 grid.

Usage:
    python scripts/make_samloss_figure_split.py
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

GROUPS = [
    ("resnet", [
        ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet-18, s = 0.995"),
        ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet-18, s = 0.999"),
    ]),
    ("vgg", [
        ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG-16, s = 0.999"),
        ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG-16, s = 0.9995"),
    ]),
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


for name, configs in GROUPS:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=False)
    for ax, (config, title) in zip(axes, configs):
        for sam_flag, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
            epochs, mean, std = mean_std(config, sam_flag, "SAM Loss/test")
            ax.plot(epochs, mean, marker="o", markersize=3, linewidth=1.8, color=color, label=label, zorder=3)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, zorder=2, linewidth=0)
        for re in ROUND_EPOCHS:
            ax.axvline(re, color="#8C8C8C", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1, alpha=0.75)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("epoch")
        style_axes(ax)
    axes[0].set_ylabel(r"$L_{\mathrm{SAM}}$ (log scale)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle(
        r"Sharpness diagnostic ($L_{\mathrm{SAM}}$) through training" + "\n"
        "(mean ± std across 3 seeds; dashed lines mark pruning rounds)",
        fontsize=11.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"sam_loss_{name}_cifar10.{ext}")
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.close(fig)

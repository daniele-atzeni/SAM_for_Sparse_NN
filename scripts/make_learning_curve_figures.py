"""Learning-curve figures for the wall x short-recovery sweep (the
strongest result in the campaign, see TODO.md).

Reads results/learning_curves.json (scp'd from the server: per-epoch
Accuracy/train and Accuracy/test, all 3 seeds, SAM vs SGD, for the 4
wall x short-recovery configs) and produces:
  figures/test_accuracy_curves.png  -- the headline metric: test accuracy
      over training, mean +/- std across 3 seeds, both architectures.
  figures/train_accuracy_curves.png -- same, for train accuracy. Shows the
      capacity-wall / implicit-regularization story from TODO.md directly:
      SGD keeps climbing toward ~100% train accuracy (comfortably
      over-parameterized) while SAM's plateaus lower at the same sparsity.

Usage:
    python scripts/make_learning_curve_figures.py
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

CONFIGS = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18, s = 0.999"),
    ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG16, s = 0.999"),
    ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG16, s = 0.9995"),
]

with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    DATA = json.load(f)


def mean_std(config, sam_flag, tag):
    """Stack the tag's series across all 3 seeds (aligned by step, since
    every run logs the same epochs) and return (epochs, mean, std)."""
    per_seed = []
    epochs_ref = None
    for seed in SEEDS:
        rows = DATA[config][seed][sam_flag][tag]
        rows = sorted(rows, key=lambda r: r["step"])
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


def make_figure(tag, ylabel, title, out_name, pct_axis=True, legend_loc="lower right"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    for ax, (config, panel_title) in zip(axes.flat, CONFIGS):
        for sam_flag, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
            epochs, mean, std = mean_std(config, sam_flag, tag)
            ax.plot(epochs, mean, linewidth=1.8, color=color, label=label, zorder=3)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, zorder=2, linewidth=0)

        for re in ROUND_EPOCHS:
            ax.axvline(re, color="#8C8C8C", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1, alpha=0.75)

        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        if pct_axis:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style_axes(ax)

    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    axes[0, 0].legend(frameon=False, loc=legend_loc, fontsize=10)
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.subplots_adjust(top=0.86, hspace=0.32)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"{out_name}.{ext}")
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.close(fig)


make_figure(
    "Accuracy/test",
    "test accuracy",
    "Test accuracy through training, tight-recovery schedule\n"
    "(mean ± std across 3 seeds; dashed lines mark the 11 pruning rounds)",
    "test_accuracy_curves",
)

make_figure(
    "Accuracy/train",
    "train accuracy",
    "Train accuracy through training, tight-recovery schedule\n"
    "(mean ± std across 3 seeds; dashed lines mark the 11 pruning rounds)",
    "train_accuracy_curves",
)

make_figure(
    "Loss/test",
    "test loss",
    "Test loss through training, tight-recovery schedule\n"
    "(mean ± std across 3 seeds; dashed lines mark the 11 pruning rounds)",
    "test_loss_curves",
    pct_axis=False,
    legend_loc="upper right",
)

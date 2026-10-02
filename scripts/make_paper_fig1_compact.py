"""Single-column version of the per-architecture accuracy figures for the
appendix (make_paper_fig1_replacement.py built these for a full
double-column spread; squeezed into one narrow appendix column, the
below-figure legend was getting pushed off the page). Same content, three
panels stacked vertically instead of 1+2 side-by-side, legend placed
inside the top panel so it can't be orphaned.

Usage:
    python scripts/make_paper_fig1_compact.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = ["13", "42", "97"]

with open(os.path.join(RESULTS_DIR, "dense_curves.json")) as f:
    DENSE = json.load(f)
with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    SPARSE = json.load(f)

SAM_SHADES = ["#8FCFC0", "#0B8A73"]
SGD_SHADES = ["#E8B08C", "#C1571E"]

PANELS = [
    ("ResNet18", "ResNet18", ["ResNet18_CIFAR10_s0.999_shortrecovery"], ["dense", "s = 0.999"]),
    ("vgg16_bn", "VGG16", ["VGG16_CIFAR10_s0.9995_shortrecovery"], ["dense", "s = 0.9995"]),
]


def mean_series(source, sam_flag, tag):
    per_seed, epochs_ref = [], None
    for seed in SEEDS:
        rows = sorted(source[seed][sam_flag][tag], key=lambda r: r["step"])
        epochs = [r["step"] for r in rows]
        values = [r["value"] for r in rows]
        if epochs_ref is None:
            epochs_ref = epochs
        per_seed.append(values)
    arr = np.array(per_seed)
    return np.array(epochs_ref), arr.mean(axis=0)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.05, 1.0)


for arch_key, arch_title, sparse_configs, labels in PANELS:
    fig, (ax_full, ax_early, ax_late) = plt.subplots(3, 1, figsize=(3.4, 7.2))

    series_sam, series_sgd = [], []
    ep, v = mean_series(DENSE[arch_key], "True", "Accuracy/test")
    series_sam.append((ep, v))
    ep, v = mean_series(DENSE[arch_key], "False", "Accuracy/test")
    series_sgd.append((ep, v))
    for cfg in sparse_configs:
        ep, v = mean_series(SPARSE[cfg], "True", "Accuracy/test")
        series_sam.append((ep, v))
        ep, v = mean_series(SPARSE[cfg], "False", "Accuracy/test")
        series_sgd.append((ep, v))

    for ax in (ax_full, ax_early, ax_late):
        for i, (label, (ep, v)) in enumerate(zip(labels, series_sam)):
            ax.plot(ep, v, color=SAM_SHADES[i], linewidth=1.3, zorder=3,
                     label=f"SAM {label}" if ax is ax_full else None)
        for i, (label, (ep, v)) in enumerate(zip(labels, series_sgd)):
            ax.plot(ep, v, color=SGD_SHADES[i], linewidth=1.3, zorder=2,
                     label=f"SGD {label}" if ax is ax_full else None)
        style_axes(ax)
        ax.tick_params(labelsize=7.5)

    ax_early.set_xlim(0, 50)
    ax_late.set_xlim(80, 180)
    ax_full.set_title(f"{arch_title}, full training", fontsize=9.5, fontweight="bold")
    ax_early.set_title("epochs 0-50", fontsize=8.5)
    ax_late.set_title("epochs 80-180", fontsize=8.5)
    ax_late.set_xlabel("epoch", fontsize=8)
    for ax in (ax_full, ax_early, ax_late):
        ax.set_ylabel("test acc.", fontsize=8)

    ax_full.legend(frameon=False, loc="lower right", fontsize=6.5)

    fig.suptitle(
        f"{arch_title}: test accuracy under iterative\npruning, dense + wall x short-recovery",
        fontsize=8.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"paper_fig1_{arch_key}_compact.{ext}")
        fig.savefig(out, dpi=220)
        print(f"Saved {out}")
    plt.close(fig)

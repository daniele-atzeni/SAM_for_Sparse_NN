"""Direct replacement for the old paper's Figure 1 / Figure 2 (test accuracy
over the sparse training phase, multiple pruning ratios overlaid, dense
baseline included). The originals (icml2026_sharpness.pdf, the NeurIPS 2026
submission) were generated with the pre-bug-fix pipeline (the unbounded
pruning loop and the hardcoded pruned=False Hessian bug) -- this rebuilds
the same visual format from the verified, bug-fixed wall x short-recovery
campaign.

Format mirrors the original: one figure per architecture, a full-range top
panel plus two zoomed bottom sub-panels (early epochs, late epochs where
the pruning rounds and final recovery live), dense + sparsity levels
overlaid, SAM vs SGD as the two color families, mean across 3 seeds (not
single noisy runs, since we have proper seed replication).

Only covers what the redone campaign actually has verified data for:
ResNet18 and VGG16 on CIFAR-10, iterative/early-stage pruning. No
WideResNet, no CIFAR-100, no pruning-finetuning regime -- those need new
experiments, not a replotting of old data.

Usage:
    python scripts/make_paper_fig1_replacement.py
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

# SAM: teal family (light -> dark = dense -> higher sparsity)
# SGD: rust family (light -> dark = dense -> higher sparsity)
SAM_SHADES = ["#8FCFC0", "#39A98D", "#0B8A73"]
SGD_SHADES = ["#E8B08C", "#D98A4E", "#C1571E"]

PANELS = [
    ("ResNet18", "ResNet18", ["ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18_CIFAR10_s0.999_shortrecovery"], ["dense", "s = 0.995", "s = 0.999"]),
    ("vgg16_bn", "VGG16", ["VGG16_CIFAR10_s0.999_shortrecovery", "VGG16_CIFAR10_s0.9995_shortrecovery"], ["dense", "s = 0.999", "s = 0.9995"]),
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
    fig = plt.figure(figsize=(10, 8.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.32, wspace=0.18)
    ax_full = fig.add_subplot(gs[0, :])
    ax_early = fig.add_subplot(gs[1, 0])
    ax_late = fig.add_subplot(gs[1, 1])

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
            ax.plot(ep, v, color=SAM_SHADES[i], linewidth=1.6, zorder=3,
                     label=f"SAM {label}" if ax is ax_full else None)
        for i, (label, (ep, v)) in enumerate(zip(labels, series_sgd)):
            ax.plot(ep, v, color=SGD_SHADES[i], linewidth=1.6, zorder=2,
                     label=f"SGD {label}" if ax is ax_full else None)
        style_axes(ax)

    ax_early.set_xlim(0, 50)
    ax_late.set_xlim(80, 180)
    ax_full.set_title(f"{arch_title}, CIFAR-10 -- full training", fontsize=11, fontweight="bold")
    ax_early.set_title("epochs 0-50", fontsize=10)
    ax_late.set_title("epochs 80-180", fontsize=10)
    for ax in (ax_full, ax_early, ax_late):
        ax.set_xlabel("epoch")
    ax_full.set_ylabel("test accuracy")
    ax_early.set_ylabel("test accuracy")

    handles, leg_labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"{arch_title}: test accuracy under iterative pruning, dense + wall x short-recovery sparsities\n"
        "(bug-fixed pipeline; mean across 3 seeds, replaces the pre-fix Figure in the NeurIPS submission)",
        fontsize=10.5, y=0.985,
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.09, right=0.97)

    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"paper_fig1_{arch_key}.{ext}")
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.close(fig)

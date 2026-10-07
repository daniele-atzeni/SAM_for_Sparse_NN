"""Paper Figure 1, compact version: test accuracy under iterative pruning,
one row of five panels at full text width, y-axis restricted to the range
where SAM and SGD differ (the first epochs, where both climb from 10%, are
clipped; the caption says so).

Usage:
    python scripts/make_fig1_compact.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
SEEDS = ["13", "42", "97"]
Y_MIN = 0.45  # lowest post-cut band edge is 0.48 (ResNet-18 SGD, s = 0.9995)

DATA = {}
for name in ["learning_curves.json", "isocompute_s0.9995.json"]:
    with open(os.path.join("results", name)) as f:
        DATA.update(json.load(f))

PANELS = [("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet-18, $s = 0.995$"),
          ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet-18, $s = 0.999$"),
          ("ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet-18, $s = 0.9995$"),
          ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG-16, $s = 0.999$"),
          ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG-16, $s = 0.9995$")]


def mean_std(config, sam_flag):
    per_seed, epochs_ref = [], None
    for seed in SEEDS:
        rows = sorted(DATA[config][seed][sam_flag]["Accuracy/test"], key=lambda r: r["step"])
        if epochs_ref is None:
            epochs_ref = [r["step"] for r in rows]
        per_seed.append([r["value"] for r in rows])
    arr = np.array(per_seed)
    return np.array(epochs_ref), arr.mean(axis=0), arr.std(axis=0)


plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7})
fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.65), sharey=True)
for ax, (config, title) in zip(axes, PANELS):
    for re in ROUND_EPOCHS:
        ax.axvline(re, color="#B0B0B0", linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    for sam_flag, label, color in [("False", "SGD", SGD_COLOR), ("True", "SAM", SAM_COLOR)]:
        epochs, mean, std = mean_std(config, sam_flag)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18, zorder=2, linewidth=0)
        ax.plot(epochs, mean, linewidth=1.1, color=color, label=label, zorder=3)
    ax.set_title(title, pad=3)
    ax.set_ylim(Y_MIN, 0.96)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 60, 120, 180])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("epoch", labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)
axes[0].set_ylabel("test accuracy")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[::-1], labels[::-1], frameon=False, loc="lower right", handlelength=1.4,
               borderaxespad=0.2)
fig.tight_layout(pad=0.3, w_pad=0.6)
for ext in ("pdf", "png"):
    out = os.path.join("figures", f"test_acc_compact.{ext}")
    fig.savefig(out, dpi=300)
    print("saved", out)

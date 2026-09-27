"""Confront training compute against final accuracy: dense vs sparse (at
each architecture's capacity-wall sparsity) x SGD vs SAM.

Per-step FLOPs (forward+backward, batch=128) were measured directly against
the real model code with torch.utils.flop_counter.FlopCounterMode:
  ResNet18:  SGD  426.112 GFLOPs/step   SAM  852.224 GFLOPs/step (exactly 2x
             -- confirmed from src/train/training.py: SAM runs a full
             second model(data)+backward at the perturbed weights)
  vgg16_bn:  SGD  254.609 GFLOPs/step   SAM  509.218 GFLOPs/step

Pruning here is unstructured masking (prune.global_unstructured /
prune.custom_from_mask): a dense tensor gets zeroed entries, but every
matmul still runs at full size -- no training FLOPs are saved by sparsity
itself in this implementation. So at a fixed epoch count, dense and sparse
cost IDENTICAL total FLOPs for a given optimizer; the entire compute
asymmetry in the whole campaign is SAM's 2x-per-step, not pruning.

Produces figures/compute_vs_accuracy.{png,pdf}: two panels (ResNet18,
VGG16), each a 2-row grouped bar chart -- final test accuracy on top,
total training PFLOPs (log scale) below -- for {Dense, Sparse @ wall
sparsity} x {SGD, SAM}, so the read is literal: how much accuracy did
that extra compute buy, and did pruning cost any training compute at all
(it didn't -- the bottom-row bars for Dense and Sparse are identical
height within an optimizer).

Usage:
    python scripts/make_compute_figure.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Measured directly, see docstring.
PER_STEP_GFLOPS = {
    "ResNet18": {"False": 426.112, "True": 852.224},
    "vgg16_bn": {"False": 254.609, "True": 509.218},
}
STEPS_PER_EPOCH = 50000 / 128  # CIFAR-10 train set, batch 128
EPOCHS = 180

with open(os.path.join(RESULTS_DIR, "dense_curves.json")) as f:
    DENSE = json.load(f)
with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    SPARSE = json.load(f)

SEEDS = ["13", "42", "97"]

ARCH_PANELS = [
    ("ResNet18", "ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18 (sparse: s = 0.999)"),
    ("vgg16_bn", "VGG16_CIFAR10_s0.9995_shortrecovery", "VGG16 (sparse: s = 0.9995)"),
]


def final_acc(source, key, seed, sam_flag):
    rows = sorted(source[key][seed][sam_flag]["Accuracy/test"], key=lambda r: r["step"])
    return rows[-1]["value"] * 100


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col", height_ratios=[2.2, 1])

groups = ["Dense", f"Sparse\n(wall)"]
x = np.arange(len(groups))
width = 0.32

for col, (arch_key, sparse_config, title) in enumerate(ARCH_PANELS):
    ax_acc = axes[0, col]
    ax_flop = axes[1, col]

    for offset, (sam_flag, label, color) in zip(
        [-1, 1], [("False", "SGD", SGD_COLOR), ("True", "SAM", SAM_COLOR)]
    ):
        dense_accs = [final_acc(DENSE, arch_key, s, sam_flag) for s in SEEDS]
        sparse_accs = [final_acc(SPARSE, sparse_config, s, sam_flag) for s in SEEDS]
        means = [np.mean(dense_accs), np.mean(sparse_accs)]
        xs = x + offset * width / 2
        ax_acc.bar(xs, means, width=width, color=color, alpha=0.65, label=label, zorder=2)
        for xi, accs in zip(xs, [dense_accs, sparse_accs]):
            jitter = np.linspace(-0.05, 0.05, len(accs))
            ax_acc.scatter(xi + jitter, accs, color="#2B2B2B", s=18, zorder=3, edgecolor="white", linewidth=0.5)

        total_pflops = PER_STEP_GFLOPS[arch_key][sam_flag] * STEPS_PER_EPOCH * EPOCHS / 1000
        ax_flop.bar(xs, [total_pflops, total_pflops], width=width, color=color, alpha=0.65, zorder=2)

    ax_acc.set_title(title, fontsize=11.5, fontweight="bold")
    ax_acc.set_xticks(x)
    style_axes(ax_acc)
    ax_flop.set_xticks(x)
    ax_flop.set_xticklabels(groups, fontsize=10)
    ax_flop.set_ylim(0, None)
    style_axes(ax_flop)

axes[0, 0].set_ylabel("final test accuracy (%)")
axes[1, 0].set_ylabel("total training\nPFLOPs")
axes[0, 0].legend(frameon=False, loc="lower left", fontsize=10)

fig.suptitle(
    "Accuracy vs. training compute: dense and sparse cost the same FLOPs at a\n"
    "fixed epoch count (unstructured pruning skips no compute) — SAM's 2x-per-step\n"
    "cost is the only real compute asymmetry in the whole comparison",
    fontsize=11,
    y=0.985,
)
fig.tight_layout(rect=[0, 0, 1, 1.0])
fig.subplots_adjust(top=0.80, hspace=0.12)

for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"compute_vs_accuracy.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
plt.close(fig)

# Also print the raw numbers as a table for THEORY_NOTES.md / TODO.md.
print("\n| arch | setting | optimizer | mean acc (%) | total PFLOPs |")
print("|---|---|---|---|---|")
for arch_key, sparse_config, _ in ARCH_PANELS:
    for setting_name, source, key in [("dense", DENSE, arch_key), ("sparse (wall)", SPARSE, sparse_config)]:
        for sam_flag, opt_label in [("False", "SGD"), ("True", "SAM")]:
            accs = [final_acc(source, key, s, sam_flag) for s in SEEDS]
            pflops = PER_STEP_GFLOPS[arch_key][sam_flag] * STEPS_PER_EPOCH * EPOCHS / 1000
            print(f"| {arch_key} | {setting_name} | {opt_label} | {np.mean(accs):.2f} | {pflops:.0f} |")

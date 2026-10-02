"""The headline iso-compute result: does SGD close the SAM-SGD gap when
given SAM's actual FLOP budget (2x epochs) instead of matched epoch count?

Answer differs by sparsity:
  s=0.999:  yes, essentially -- SGD@360ep (mean 81.56) matches/slightly
            exceeds SAM@180ep (mean 81.03). The epoch-matched gap there is
            substantially a compute-budget artifact.
  s=0.9995: no -- SGD@360ep (mean 68.23) recovers only ~2pp of the 6.4pp
            gap, leaving SAM@180ep ahead by ~4.4pp even at equal compute.
            This is the first result in the whole campaign where SAM's
            edge survives giving SGD its FLOPs back.

Usage:
    python scripts/make_isocompute_figure.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
SGD_ISO_COLOR = "#8C4A1E"  # darker rust: SGD given SAM's compute budget

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = ["13", "42", "97"]

with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    LC = json.load(f)
with open(os.path.join(RESULTS_DIR, "isocompute_s0.9995.json")) as f:
    ISO_9995 = json.load(f)
with open(os.path.join(RESULTS_DIR, "isocompute_s0.999.json")) as f:
    ISO_999 = json.load(f)


def final_acc(series):
    return sorted(series["Accuracy/test"], key=lambda r: r["step"])[-1]["value"] * 100


groups = [
    ("s = 0.999", "ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18_CIFAR10_s0.999_shortrecovery_isocompute_sgd", ISO_999),
    ("s = 0.9995", "ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd", ISO_9995),
]

bar_labels = ["SGD\n@180ep", "SGD\n@360ep\n(iso-compute)", "SAM\n@180ep"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)

for ax, (title, normal_cfg, iso_cfg, iso_source) in zip(axes, groups):
    normal_source = LC[normal_cfg] if normal_cfg in LC else ISO_9995[normal_cfg]

    sgd_180 = [final_acc(normal_source[s]["False"]) for s in SEEDS]
    sam_180 = [final_acc(normal_source[s]["True"]) for s in SEEDS]
    sgd_360 = [final_acc(iso_source[iso_cfg][s]["False"]) for s in SEEDS]

    data = [sgd_180, sgd_360, sam_180]
    colors = [SGD_COLOR, SGD_ISO_COLOR, SAM_COLOR]
    x = np.arange(3)
    means = [np.mean(d) for d in data]
    ax.bar(x, means, color=colors, alpha=0.7, width=0.6, zorder=2)
    for xi, vals in zip(x, data):
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(xi + jitter, vals, color="#2B2B2B", s=26, zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9.5)
    ax.set_title(f"ResNet18, {title}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("final test accuracy (%)")
fig.suptitle(
    "Does SGD close the gap with SAM's own compute budget?\n"
    "Yes at s=0.999 (SGD@360ep matches SAM@180ep) -- no at s=0.9995 (SAM still ahead by ~4pp)",
    fontsize=11.5, y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])

for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"isocompute_summary.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
plt.close(fig)

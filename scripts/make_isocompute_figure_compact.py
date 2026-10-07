"""Single-column version of the iso-compute summary figure, for the main
text. Panels stacked vertically instead of side-by-side, sized for a
single AISTATS column (~3.3in wide) rather than a full-width figure*.

Usage:
    python scripts/make_isocompute_figure_compact.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
SGD_ISO_COLOR = "#8C4A1E"

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
# Final models re-evaluated on the full 10,000-image test set
# (scripts/analyze_existing_checkpoints.py); the training logs only cover
# the first 1,280 test images.
with open(os.path.join(RESULTS_DIR, "existing_checkpoints_analysis.json")) as f:
    FULL = json.load(f)["full_test_accuracy"]


def full_acc(cfg, sam_flag):
    return [FULL[cfg][sam_flag][s]["full"] * 100 for s in SEEDS]


def final_acc(series):
    return sorted(series["Accuracy/test"], key=lambda r: r["step"])[-1]["value"] * 100


groups = [
    ("s = 0.999", "ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18_CIFAR10_s0.999_shortrecovery_isocompute_sgd", ISO_999),
    ("s = 0.9995", "ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd", ISO_9995),
]

bar_labels = ["SGD\n180ep", "SGD\n360ep", "SAM\n180ep"]
fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.4), sharex=True)

for ax, (title, normal_cfg, iso_cfg, iso_source) in zip(axes, groups):
    normal_source = LC[normal_cfg] if normal_cfg in LC else ISO_9995[normal_cfg]

    sgd_180 = full_acc(normal_cfg, "False")
    sam_180 = full_acc(normal_cfg, "True")
    sgd_360 = full_acc(iso_cfg, "False")

    data = [sgd_180, sgd_360, sam_180]
    colors = [SGD_COLOR, SGD_ISO_COLOR, SAM_COLOR]
    x = np.arange(3)
    means = [np.mean(d) for d in data]
    for xi, m, c in zip(x, means, colors):
        ax.hlines(m, xi - 0.25, xi + 0.25, color=c, linewidth=3, zorder=2)
    for xi, vals in zip(x, data):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(xi + jitter, vals, color="#2B2B2B", s=14, zorder=3, edgecolor="white", linewidth=0.4)
    for xi, m, vals in zip(x, means, data):
        ax.text(xi, max(max(vals), m) + 0.6, f"{m:.1f}", ha="center", va="bottom", fontsize=7)
    lo, hi = min(min(d) for d in data), max(max(d) for d in data)
    ax.set_ylim(lo - 2, hi + 2.5)

    ax.set_xticks(x)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylabel("test acc. (%)", fontsize=8)

import matplotlib.patches as mpatches


def _no_suptitle(*args, **kwargs):
    """Paper figures carry no in-figure title; the LaTeX caption describes them."""
handles = [
    mpatches.Patch(color=SGD_COLOR, alpha=0.75, label="SGD, 180ep"),
    mpatches.Patch(color=SGD_ISO_COLOR, alpha=0.75, label="SGD, 360ep"),
    mpatches.Patch(color=SAM_COLOR, alpha=0.75, label="SAM, 180ep"),
]
axes[1].set_xticklabels(bar_labels, fontsize=7.5)
for ax in axes:
    ax.set_xlabel("")
_no_suptitle(
    "Does SGD close the gap given\nSAM's own compute budget?",
    fontsize=9.5, y=0.99,
)
fig.subplots_adjust(top=0.93, bottom=0.1, left=0.17, right=0.97, hspace=0.45)

for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"isocompute_summary_compact.{ext}")
    fig.savefig(out, dpi=220)
    print(f"Saved {out}")
plt.close(fig)

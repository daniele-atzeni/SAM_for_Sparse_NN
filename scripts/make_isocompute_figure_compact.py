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


def final_acc(series):
    return sorted(series["Accuracy/test"], key=lambda r: r["step"])[-1]["value"] * 100


groups = [
    ("s = 0.999 (closer to wall)", "ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18_CIFAR10_s0.999_shortrecovery_isocompute_sgd", ISO_999),
    ("s = 0.9995 (most extreme)", "ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet18_CIFAR10_s0.9995_shortrecovery_isocompute_sgd", ISO_9995),
]

bar_labels = ["SGD\n180ep", "SGD\n360ep", "SAM\n180ep"]
fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.4), sharex=True)

for ax, (title, normal_cfg, iso_cfg, iso_source) in zip(axes, groups):
    normal_source = LC[normal_cfg] if normal_cfg in LC else ISO_9995[normal_cfg]

    sgd_180 = [final_acc(normal_source[s]["False"]) for s in SEEDS]
    sam_180 = [final_acc(normal_source[s]["True"]) for s in SEEDS]
    sgd_360 = [final_acc(iso_source[iso_cfg][s]["False"]) for s in SEEDS]

    data = [sgd_180, sgd_360, sam_180]
    colors = [SGD_COLOR, SGD_ISO_COLOR, SAM_COLOR]
    x = np.arange(3)
    means = [np.mean(d) for d in data]
    ax.bar(x, means, color=colors, alpha=0.75, width=0.6, zorder=2)
    for xi, vals in zip(x, data):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(xi + jitter, vals, color="#2B2B2B", s=14, zorder=3, edgecolor="white", linewidth=0.4)
    for xi, m, vals in zip(x, means, data):
        ax.text(xi, max(max(vals), m) + 2.5, f"{m:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(max(d) for d in data) + 10)

    ax.set_xticks(x)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylabel("test acc. (%)", fontsize=8)

import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color=SGD_COLOR, alpha=0.75, label="SGD, 180ep"),
    mpatches.Patch(color=SGD_ISO_COLOR, alpha=0.75, label="SGD, 360ep"),
    mpatches.Patch(color=SAM_COLOR, alpha=0.75, label="SAM, 180ep"),
]
axes[0].legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False)
axes[1].set_xticklabels(["", "", ""])
for ax in axes:
    ax.set_xlabel("")
fig.suptitle(
    "Does SGD close the gap given\nSAM's own compute budget?",
    fontsize=9.5, y=0.99,
)
fig.subplots_adjust(top=0.89, bottom=0.06, left=0.17, right=0.97, hspace=0.45)

for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"isocompute_summary_compact.{ext}")
    fig.savefig(out, dpi=220)
    print(f"Saved {out}")
plt.close(fig)

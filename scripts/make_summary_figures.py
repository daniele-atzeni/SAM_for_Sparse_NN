"""Two more figures from results/learning_curves.json:

  figures/hessian_trace_curves.{png,pdf} -- Hessian trace, SAM vs SGD, over
      full training. This is "the one fully robust finding" from the
      strong-pruning sweep (TODO.md): SAM's trace runs consistently 3-10x
      below SGD's. ResNet18 only -- VGG16's Hessian estimator is unreliable
      at these sparsities (trace spikes to 1e6-1e12 right after the last
      cut in several seeds, the same estimator issue documented in
      THEORY_NOTES.md Sec. 7 for lambda1; plotting it would mislead rather
      than inform), so it's excluded here rather than shown misleadingly.
  figures/final_accuracy_gap_summary.{png,pdf} -- headline result: final
      test-accuracy gap (SAM-SGD) at epoch 180, all 4 wall x short-recovery
      configs, mean bar + individual seed points.

Usage:
    python scripts/make_summary_figures.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
GAP_COLOR = "#1F5FA6"
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
SEEDS = ["13", "42", "97"]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "learning_curves.json")) as f:
    DATA = json.load(f)


def series(config, seed, sam_flag, tag):
    rows = sorted(DATA[config][seed][sam_flag][tag], key=lambda r: r["step"])
    return [r["step"] for r in rows], [r["value"] for r in rows]


def mean_std(config, sam_flag, tag):
    per_seed, epochs_ref = [], None
    for seed in SEEDS:
        epochs, values = series(config, seed, sam_flag, tag)
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


# ---------------------------------------------------------------------------
# Figure: Hessian trace, ResNet18 only
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)

trace_configs = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18, s = 0.999"),
]
for ax, (config, title) in zip(axes, trace_configs):
    for sam_flag, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        epochs, mean, std = mean_std(config, sam_flag, "trace/test")
        ax.plot(epochs, mean, linewidth=1.8, color=color, label=label, zorder=3)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, zorder=2, linewidth=0)
    for re in ROUND_EPOCHS:
        ax.axvline(re, color="#8C8C8C", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1, alpha=0.75)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch")
    style_axes(ax)

axes[0].set_ylabel(r"Hessian trace (log scale)")
axes[0].legend(frameon=False, loc="upper right", fontsize=10)
fig.suptitle(
    "SAM's flat-minima trace advantage holds throughout training\n"
    "(mean ± std across 3 seeds; dashed lines mark the 11 pruning rounds)",
    fontsize=11,
    y=0.985,
)
fig.tight_layout(rect=[0, 0, 1, 1.0])
fig.subplots_adjust(top=0.83)
for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"hessian_trace_curves.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: final-accuracy-gap headline summary
# ---------------------------------------------------------------------------
summary_configs = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18\ns = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18\ns = 0.999"),
    ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG16\ns = 0.999"),
    ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG16\ns = 0.9995"),
]

gaps_per_config = []
for config, _ in summary_configs:
    gaps = []
    for seed in SEEDS:
        _, sam_vals = series(config, seed, "True", "Accuracy/test")
        _, sgd_vals = series(config, seed, "False", "Accuracy/test")
        gaps.append((sam_vals[-1] - sgd_vals[-1]) * 100)
    gaps_per_config.append(gaps)

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(summary_configs))
means = [np.mean(g) for g in gaps_per_config]
bar_colors = [SAM_COLOR if m > 0 else SGD_COLOR for m in means]
ax.bar(x, means, color=bar_colors, alpha=0.55, width=0.55, zorder=2)
for xi, gaps in zip(x, gaps_per_config):
    jitter = np.linspace(-0.09, 0.09, len(gaps))
    ax.scatter(xi + jitter, gaps, color="#2B2B2B", s=28, zorder=3, edgecolor="white", linewidth=0.6)

ax.axhline(0, color="#2B2B2B", linewidth=1.0, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in summary_configs], fontsize=10)
ax.set_ylabel("final test-accuracy gap, SAM − SGD (pp)")
style_axes(ax)
ax.grid(False, axis="x")
ax.set_title(
    "SAM's final-accuracy edge grows with sparsity\n"
    "(bars = mean across 3 seeds; dots = individual seeds)",
    fontsize=11.5,
    fontweight="bold",
)
fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(FIGURES_DIR, f"final_accuracy_gap_summary.{ext}")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
plt.close(fig)

"""Figures for THEORY_NOTES.md, from the telescoping-bound checkpoint analysis.

Reads results/telescoping_*.json (scp'd from the server) and produces:
  figures/lambda1_relaxation.png  -- the central finding (Sec. 4): does
      lambda1 relax within a 15-epoch round, SAM vs SGD, ResNet18 only
      (VGG16's eigenvalue estimate is unreliable at this scale, see
      THEORY_NOTES.md Sec. 7 -- not plotted).
  figures/grad_norm_trajectory.png -- Prop 3.1' residual over full
      training, all 4 measured configs (both architectures).

Usage:
    python scripts/make_theory_figures.py
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load(name):
    path = os.path.join(RESULTS_DIR, f"telescoping_{name}_seed13.json")
    return json.load(open(path))


def series(data, key, field):
    rows = [r for r in data[key] if field in r]
    rows.sort(key=lambda r: r["epoch"])
    return [r["epoch"] for r in rows], [r[field] for r in rows]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def mark_round(ax, epoch):
    ax.axvline(epoch, color="#8C8C8C", linewidth=1.1, linestyle=(0, (4, 2)), zorder=1, alpha=0.9)


# ---------------------------------------------------------------------------
# Figure 1: lambda1 within-round relaxation, ResNet18 only, epochs 133-167
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)

configs = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18, s = 0.999"),
]

for ax, (name, title) in zip(axes, configs):
    data = load(name)
    for key, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        ep, l1 = series(data, key, "lambda1")
        ep = [e for e in ep if 133 <= e <= 167]
        l1 = [v for e, v in zip(*series(data, key, "lambda1")) if 133 <= e <= 167]
        ax.plot(ep, l1, marker="o", markersize=5, linewidth=2, color=color, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_xticks([135, 140, 145, 150, 155, 160, 165])
    style_axes(ax)

    for re in [135, 150, 165]:
        mark_round(ax, re)

axes[0].set_ylabel(r"$\lambda_1(H_m)$ (log scale)")
axes[0].legend(frameon=False, loc="upper left", fontsize=10)

fig.suptitle(
    "Within-round curvature relaxation: SAM settles, SGD often doesn't\n"
    "(dashed lines = pruning rounds 9/10/11, epochs 135/150/165 — each gets a 15-epoch recovery window)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
for ext in ("png", "pdf"):
    out1 = os.path.join(FIGURES_DIR, f"lambda1_relaxation.{ext}")
    fig.savefig(out1, dpi=200)
    print(f"Saved {out1}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: grad_norm over full training, all 4 configs
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

configs2 = [
    ("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet18, s = 0.995"),
    ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet18, s = 0.999"),
    ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG16, s = 0.999"),
    ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG16, s = 0.9995"),
]

for ax, (name, title) in zip(axes.flat, configs2):
    data = load(name)
    for key, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        ep, gn = series(data, key, "grad_norm")
        ax.plot(ep, gn, linewidth=1.8, color=color, label=label, zorder=3)

    for re in ROUND_EPOCHS:
        ax.axvline(re, color="#8C8C8C", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1, alpha=0.75)

    ax.set_title(title, fontsize=11, fontweight="bold")
    style_axes(ax)

for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel(r"restricted grad. norm $\|P_m^T\nabla L\|$")

axes[0, 0].legend(frameon=False, loc="upper right", fontsize=10)
fig.suptitle(
    "Restricted gradient norm through training, tight-recovery schedule\n"
    "(vertical lines mark the 11 pruning rounds, every 15 epochs)",
    fontsize=12,
    y=0.985,
)
fig.tight_layout(rect=[0, 0, 1, 1.0])
fig.subplots_adjust(top=0.86, hspace=0.32)
for ext in ("png", "pdf"):
    out2 = os.path.join(FIGURES_DIR, f"grad_norm_trajectory.{ext}")
    fig.savefig(out2, dpi=200)
    print(f"Saved {out2}")
plt.close(fig)

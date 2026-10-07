"""Paper Figure 2: what each cut does to the two models, one column per
configuration.
Top: error increase caused by each cut (pruned point minus the retrained
model of the previous round, BatchNorm recalibrated, seed 13; from
results/linear_connectivity_seed13.json).
Bottom: restricted gradient norm right after each cut (Proposition 3.2),
logged during training on one batch, mean over 3 seeds with min-max band
(results/post_pruning_metrics.json).

Usage:
    python scripts/make_cut_figure_two_rows.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
SEEDS = ["13", "42", "97"]
RUNS = [("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet-18, $s = 0.995$"),
        ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet-18, $s = 0.999$"),
        ("ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet-18, $s = 0.9995$"),
        ("VGG16_CIFAR10_s0.999_shortrecovery", "VGG-16, $s = 0.999$"),
        ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG-16, $s = 0.9995$")]
OPTS = [("False", "SGD", SGD_COLOR), ("True", "SAM", SAM_COLOR)]

conn = json.load(open(os.path.join("results", "linear_connectivity_seed13.json")))
grad = json.load(open(os.path.join("results", "post_pruning_metrics.json")))

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7})
fig, axes = plt.subplots(2, 5, figsize=(7.0, 2.9), sharex=True, sharey="row")
for col, (run, title) in enumerate(RUNS):
    top, bot = axes[0, col], axes[1, col]
    for sam, label, color in OPTS:
        rec = conn[f"{run}|{sam}"]["recovery"]
        cuts = [r["cut"] for r in rec]
        pruned = [100 * r["curve"][0]["err"] for r in rec]
        retrained = [100 * r["curve"][-1]["err"] for r in rec]
        inc = [p - q for p, q in zip(pruned[1:], retrained[:-1])]
        top.plot(cuts[1:], inc, color=color, marker="o", markersize=2.2, linewidth=1.1, label=label, zorder=3)

        per = [dict(grad[f"{run}|seed_{s}|SAM_{sam}"]["masked_grad_norm/post_pruning"]) for s in SEEDS]
        gc = sorted(per[0])
        vals = np.array([[p[c] for c in gc] for p in per])
        gc = np.array(gc, dtype=float)
        bot.fill_between(gc, vals.min(0), vals.max(0), color=color, alpha=0.18, linewidth=0, zorder=2)
        bot.plot(gc, vals.mean(0), color=color, marker="o", markersize=2.2, linewidth=1.1, label=label, zorder=3)
    top.axhline(0, color="#B0B0B0", linewidth=0.5, zorder=1)
    top.set_title(title, pad=3)
    bot.set_yscale("log")
    bot.set_xlabel("epoch of the cut", labelpad=1)
    for ax in (top, bot):
        ax.set_xlim(8, 172)
        ax.set_xticks([30, 90, 150])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, pad=1.5)
axes[0, 0].set_ylabel("error increase (pts)")
axes[1, 0].set_ylabel("restricted grad. norm")
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend(handles[::-1], labels[::-1], frameon=False, loc="upper left", handlelength=1.4,
                  borderaxespad=0.2)
fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.5)
fig.align_ylabels(axes[:, 0])
for ext in ("pdf", "png"):
    out = os.path.join("figures", f"cut_damage_gradient.{ext}")
    fig.savefig(out, dpi=300)
    print("saved", out)

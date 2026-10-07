"""Curvature figures for the constrained iterative-pruning runs.

  figures/hessian_trace_curves.{png,pdf}  ResNet-18 at s = 0.995, 0.999, 0.9995
  figures/sam_loss_resnet_cifar10.{png,pdf}  sharpness gap, ResNet-18, same 3 levels
  figures/sam_loss_vgg_cifar10.{png,pdf}     sharpness gap, VGG-16 at s = 0.999, 0.9995

"SAM Loss/test" is the logged sharpness gap L(theta + eps) - L(theta) with
||eps|| = rho = 0.5 along the normalized gradient. On a log axis we plot the
geometric mean over the 3 seeds with the min-max range as the band (a
mean +/- std band runs below zero on log scale).

Usage:
    python scripts/make_curvature_figures.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAM_COLOR = "#0B8A73"
SGD_COLOR = "#C1571E"
ROUND_EPOCHS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
SEEDS = ["13", "42", "97"]
FLOOR = 1e-3

DATA = {}
for name in ["learning_curves.json", "curvature_s0.9995.json"]:
    with open(os.path.join("results", name)) as f:
        DATA.update(json.load(f))


def geo_band(config, sam_flag, tag):
    per_seed, ref = [], None
    for seed in SEEDS:
        rows = sorted(DATA[config][seed][sam_flag][tag], key=lambda r: r["step"])
        if ref is None:
            ref = [r["step"] for r in rows]
        per_seed.append([max(r["value"], FLOOR) for r in rows])
    n = min(len(v) for v in per_seed)
    arr = np.array([v[:n] for v in per_seed])
    return np.array(ref[:n]), np.exp(np.log(arr).mean(0)), arr.min(0), arr.max(0)


def panel(ax, config, title, tag):
    for sam_flag, label, color in [("True", "SAM", SAM_COLOR), ("False", "SGD", SGD_COLOR)]:
        x, gm, lo, hi = geo_band(config, sam_flag, tag)
        ax.plot(x, gm, linewidth=1.6, color=color, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=2, linewidth=0)
    for re in ROUND_EPOCHS:
        ax.axvline(re, color="#8C8C8C", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("epoch")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)


def figure(configs, tag, ylabel, out_name, width):
    fig, axes = plt.subplots(1, len(configs), figsize=(width, 3.8))
    for ax, (config, title) in zip(axes, configs):
        panel(ax, config, title, tag)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join("figures", f"{out_name}.{ext}")
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.close(fig)


RESNET = [("ResNet18_CIFAR10_s0.995_shortrecovery", "ResNet-18, s = 0.995"),
          ("ResNet18_CIFAR10_s0.999_shortrecovery", "ResNet-18, s = 0.999"),
          ("ResNet18_CIFAR10_s0.9995_shortrecovery", "ResNet-18, s = 0.9995")]
VGG = [("VGG16_CIFAR10_s0.999_shortrecovery", "VGG-16, s = 0.999"),
       ("VGG16_CIFAR10_s0.9995_shortrecovery", "VGG-16, s = 0.9995")]

figure(RESNET, "trace/test", "Hessian trace", "hessian_trace_curves", 14)
figure(RESNET, "SAM Loss/test", r"$\mathcal{L}(\theta+\epsilon)-\mathcal{L}(\theta)$", "sam_loss_resnet_cifar10", 14)
figure(VGG, "SAM Loss/test", r"$\mathcal{L}(\theta+\epsilon)-\mathcal{L}(\theta)$", "sam_loss_vgg_cifar10", 9.5)

"""Shared registries and builder functions for models, datasets, optimizers, and schedulers.

This module centralises the lookup tables and construction logic that was
previously copy-pasted across every ``main_*.py`` entry point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import (
    MLP,
    # CIFAR-style ResNets (3×3 stem)
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    # ResNet-Plus (proper CIFAR depths)
    ResNet20, ResNet32, ResNet44, ResNet56, ResNet110,
    # VGG
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    # VGG-Plus
    vgg11_plus, vgg11_bn_plus, vgg13_mingze, vgg16_mingze, vgg19_mingze,
    # Wide ResNets
    WideResNet16_8, WideResNet28_10,
    WideResNet34_10_madry, WideResNet16_8_madry,
    # Vision Transformer
    ViT,
    # ImageNet ResNets (7×7 stem)
    ResNet18_IN, ResNet34_IN, ResNet50_IN, ResNet101_IN, ResNet152_IN,
)
from src.data import (
    get_mnist_loaders,
    get_fashion_mnist_loaders,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_imagenet_loaders,
)
from src.train.SAM import SAM
from src.train.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    WarmupCosineAnnealingLR,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type | callable] = {
    # MLP
    "MLP": MLP,
    # CIFAR ResNets
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    # ResNet-Plus
    "ResNet20": ResNet20,
    "ResNet32": ResNet32,
    "ResNet44": ResNet44,
    "ResNet56": ResNet56,
    "ResNet110": ResNet110,
    # VGG
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
    # VGG-Plus
    "vgg11_plus": vgg11_plus,
    "vgg11_bn_plus": vgg11_bn_plus,
    "vgg13_mingze": vgg13_mingze,
    "vgg16_mingze": vgg16_mingze,
    "vgg19_mingze": vgg19_mingze,
    # Wide ResNets
    "WideResNet16_8": WideResNet16_8,
    "WideResNet28_10": WideResNet28_10,
    "WideResNet34_10_madry": WideResNet34_10_madry,
    "WideResNet16_8_madry": WideResNet16_8_madry,
    # Vision Transformer
    "ViT": ViT,
    # ImageNet ResNets
    "ResNet18_IN": ResNet18_IN,
    "ResNet34_IN": ResNet34_IN,
    "ResNet50_IN": ResNet50_IN,
    "ResNet101_IN": ResNet101_IN,
    "ResNet152_IN": ResNet152_IN,
}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, callable] = {
    "MNIST": get_mnist_loaders,
    "fashionMNIST": get_fashion_mnist_loaders,
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
    "ImageNet": get_imagenet_loaders,
}


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_model(name: str, params: dict) -> nn.Module:
    """Instantiate a model from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**params)


def build_dataloaders(
    name: str,
    batch_size: int,
    **kwargs,
) -> tuple:
    """Return (train_loader, test_loader) for the given dataset name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](batch_size=batch_size, **kwargs)


def build_scheduler(config: dict, learning_rate: float):
    """Build a learning-rate scheduler from a config dict (may be ``None``).

    Expected keys inside *config["training"]["scheduler"]*:
        type: one of "MultiStepLR", "CosineAnnealingLR", "WarmupCosineAnnealingLR"
        (plus scheduler-specific parameters such as step_size, gamma, T_max, …)
    """
    sched_cfg = config.get("training", {}).get("scheduler", {})
    sched_type = sched_cfg.get("type")

    if sched_type == "MultiStepLR":
        return MultiStepLR(
            learning_rate,
            milestones=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )
    if sched_type == "CosineAnnealingLR":
        return CosineAnnealingLR(learning_rate, T_max=sched_cfg["T_max"])
    if sched_type == "WarmupCosineAnnealingLR":
        return WarmupCosineAnnealingLR(
            learning_rate,
            T_max=sched_cfg["T_max"],
            warmup_epochs=sched_cfg["warmup_epochs"],
        )
    return None


def build_criterion(name: str) -> nn.Module:
    """Build a loss function by name."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: '{name}'")


def build_optimizers(
    model: nn.Module,
    config: dict,
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, SAM]:
    """Build both the base optimizer and a SAM wrapper from a config dict.

    Returns (base_optimizer, sam_optimizer).
    """
    training_cfg = config["training"]
    optimizer_name = training_cfg["optimizer"]
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    rho = training_cfg.get("rho", 0.5)

    if optimizer_name == "sgd":
        momentum = training_cfg.get("momentum", 0.9)
        base_cls = optim.SGD
        base_opt = base_cls(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        sam_opt = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            base_cls,
            rho=rho,
            adaptive=False,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "adamw":
        base_cls = optim.AdamW
        base_opt = base_cls(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        sam_opt = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            base_cls,
            rho=rho,
            adaptive=False,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

    return base_opt, sam_opt

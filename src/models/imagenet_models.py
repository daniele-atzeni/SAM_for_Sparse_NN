"""Torchvision ResNets with the standard ImageNet stem (7x7 conv, stride=2, MaxPool).

These are distinct from the CIFAR-style ResNets in cifar_resnet.py, which use
a 3x3 stem and fixed avg_pool2d(out, 4) unsuitable for 224x224 inputs.
"""
import torchvision.models as tv_models

__all__ = [
    "ResNet18_IN", "ResNet34_IN", "ResNet50_IN",
    "ResNet101_IN", "ResNet152_IN",
]


def ResNet18_IN(**kwargs):
    return tv_models.resnet18(weights=None, **kwargs)


def ResNet34_IN(**kwargs):
    return tv_models.resnet34(weights=None, **kwargs)


def ResNet50_IN(**kwargs):
    return tv_models.resnet50(weights=None, **kwargs)


def ResNet101_IN(**kwargs):
    return tv_models.resnet101(weights=None, **kwargs)


def ResNet152_IN(**kwargs):
    return tv_models.resnet152(weights=None, **kwargs)

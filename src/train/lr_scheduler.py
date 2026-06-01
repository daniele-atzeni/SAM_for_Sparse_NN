"""Custom learning-rate schedulers.

Each scheduler is a callable: ``scheduler(optimizer, epoch)`` sets every
parameter-group's ``lr`` according to the schedule.
"""

from __future__ import annotations

import math
from typing import List


class StepLRforWRN:
    """Two-stage step decay used with Wide ResNets.

    Drops the learning rate by 5× at 30% of training and again at 60%.
    """

    def __init__(self, learning_rate: float, total_epochs: int):
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, optimizer, epoch):
        if epoch < self.total_epochs * 3 / 10:
            lr = self.base
        elif epoch < self.total_epochs * 6 / 10:
            lr = self.base * 0.2
        else:
            lr = self.base * 0.04  # 0.2 ** 2

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class MultiStepLR:
    """Decay the learning rate by *gamma* at each milestone epoch.

    Parameters
    ----------
    learning_rate : float
        Initial (base) learning rate.
    milestones : list[int]
        Epochs at which to multiply the current LR by *gamma*.
    gamma : float
        Multiplicative decay factor applied at each milestone.
    """

    def __init__(self, learning_rate: float, milestones: List[int], gamma: float):
        self.milestones = milestones
        self.base = learning_rate
        self.gamma = gamma

    def __call__(self, optimizer, epoch):
        lr = self.base
        for milestone in self.milestones:
            if epoch >= milestone - 1:
                lr *= self.gamma

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class CosineAnnealingLR:
    """Cosine annealing from *learning_rate* down to 0 over *T_max* epochs.

    Parameters
    ----------
    learning_rate : float
        Peak (initial) learning rate.
    T_max : int
        Total number of epochs for one cosine half-period.
    """

    def __init__(self, learning_rate: float, T_max: int):
        self.base = learning_rate
        self.T_max = T_max

    def __call__(self, optimizer, epoch):
        lr = self.base * 0.5 * (1 + math.cos(math.pi * epoch / self.T_max))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class WarmupCosineAnnealingLR:
    """Linear warmup followed by cosine annealing.

    Parameters
    ----------
    base_lr : float
        Peak learning rate reached at the end of warmup.
    T_max : int
        Total number of training epochs (warmup + cosine).
    warmup_epochs : int
        Number of epochs for the linear warmup phase.
    """

    def __init__(self, base_lr: float, T_max: int, warmup_epochs: int):
        self.base_lr = base_lr
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs

    def __call__(self, optimizer, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            t = epoch - self.warmup_epochs
            T = self.T_max - self.warmup_epochs
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * t / T))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

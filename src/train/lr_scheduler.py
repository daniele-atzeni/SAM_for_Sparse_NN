from typing import List
import math

class StepLRforWRN:
    def __init__(self, learning_rate: float, total_epochs: int):
        """_summary_

        Args:
            learning_rate (float): _description_
            total_epochs (int): _description_
        """
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, optimizer, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        # elif epoch < self.total_epochs * 8/10:
        #     lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 2

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class MultiStepLR:
    def __init__(self, learning_rate: float, milestones: List[int], gamma: float):
        """_summary_

        Args:
            learning_rate (float): _description_
            milestones (List[int]): _description_
            gamma (float): _description_
        """
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
    def __init__(self, learning_rate: float, T_max: int):
        """_summary_

        Args:
            learning_rate (float): _description_
            T_max (int): _description_
        """
        self.base = learning_rate
        self.T_max = T_max

    def __call__(self, optimizer, epoch):
        lr = self.base * 0.5 * (1 + math.cos(math.pi * epoch / self.T_max))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

import math

class WarmupCosineAnnealingLR:
    def __init__(self, base_lr: float, T_max: int, warmup_epochs: int):
        self.base_lr = base_lr
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs

    def __call__(self, optimizer, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            t = epoch - self.warmup_epochs
            T = self.T_max - self.warmup_epochs
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * t / T))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
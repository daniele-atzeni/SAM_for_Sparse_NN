import os, torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


def load(root: str = "./src/data/DATA",
         normalize: bool=True,
         randomize: bool=True,
         cutout: bool=False) -> Tuple[Dataset, Dataset]:
    """load the cifar10 dataset.
    Args:
        root (str): the root path of the dataset.
        normalize (bool): whether to normalize the dataset.
        randomize (bool): whether to randomize the dataset.
        cutout (bool): whether to apply cutout.
    Returns:
        return the dataset.
    """

    # prepare the transform
    transform_train = transforms.Compose([
        *([transforms.RandomCrop(32, padding=4), 
           transforms.RandomHorizontalFlip()] if randomize else []), 
        transforms.ToTensor(),
        *([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if normalize else []),
        *([Cutout()] if cutout else []),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        *([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if normalize else []),
    ])

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    # show basic info of dataset
    return trainset, testset


def get_cifar10_loaders(batch_size) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    trainset, testset = load()
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
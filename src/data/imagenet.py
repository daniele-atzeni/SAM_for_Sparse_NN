"""ImageNet data loaders using a torchvision ImageFolder layout.

Expected directory structure (standard torchvision convention)::

    <root>/
        train/<class_name>/<image>.JPEG
        val/<class_name>/<image>.JPEG

Use ``scripts/parquet_to_imagefolder.py`` to convert HuggingFace parquet
downloads into this layout.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_loaders(
    batch_size: int = 64,
    root: str = "./src/data/DATA/ImageNet",
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for ImageNet.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for both loaders.
    root : str
        Path to the ImageFolder root (must contain ``train/`` and ``val/``
        subdirectories).
    num_workers : int
        Number of data-loading workers per loader.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

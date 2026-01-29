from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(
        main_folder: str = './src/data/DATA', 
        batch_size: int = 128
        ) -> tuple[DataLoader, DataLoader]:
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(main_folder, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(main_folder, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_fashion_mnist_loaders(
        main_folder: str = './src/data/DATA', 
        batch_size: int = 128
        ) -> tuple[DataLoader, DataLoader]:
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(main_folder, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(main_folder, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

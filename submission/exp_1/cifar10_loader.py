"""
Simple CIFAR-10 data loader for SGLD experiment.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def load_cifar10_data(batch_size=128):
    """Load CIFAR-10 data with simple preprocessing."""
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training data
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])
    
    # Create data loaders (single worker for simplicity)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

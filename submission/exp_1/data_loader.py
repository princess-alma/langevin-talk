"""
CIFAR-10 Data Loader

This module provides utilities for loading and preprocessing CIFAR-10 data
with train/validation split, following the same pattern as the MNIST loader.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def load_cifar10_data(batch_size=128, validation_split=0.1, download=True):
    """
    Load CIFAR-10 data with train/validation split.
    
    Args:
        batch_size (int): Batch size for data loaders
        validation_split (float): Fraction of training data to use for validation
        download (bool): Whether to download the dataset if not found
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Define transforms for training (with data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Define transforms for validation and test (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load the training data
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=download,
        transform=train_transform
    )
    
    # Download and load the test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=download,
        transform=test_transform
    )
    
    # Create train/validation split
    num_train = len(full_train_dataset)
    num_val = int(num_train * validation_split)
    num_train_remaining = num_train - num_val
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [num_train_remaining, num_val],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create validation dataset with test transforms (no augmentation)
    val_dataset.dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=False,  # Already downloaded
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_info():
    """Get information about CIFAR-10 dataset."""
    return {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': (32, 32),
        'class_names': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }


def load_cifar10_simple(batch_size=4):
    """
    Simple CIFAR-10 loader (your original code with improvements).
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (trainloader, testloader)
    """
    
    # Define a transform to convert images to PyTorch tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,  # This does the magic!
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True
    )

    # Download and load the test data
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,  # Also downloads if not found
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False
    )

    print("CIFAR-10 datasets loaded successfully!")
    return trainloader, testloader


# Example usage and testing
if __name__ == "__main__":
    print("Testing CIFAR-10 data loaders...")
    
    # Test the advanced loader
    print("\n1. Testing advanced CIFAR-10 loader:")
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get dataset info
    info = get_cifar10_info()
    print(f"\nDataset info:")
    print(f"Classes: {info['num_classes']}")
    print(f"Channels: {info['input_channels']}")
    print(f"Size: {info['input_size']}")
    print(f"Class names: {info['class_names']}")
    
    # Test a batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"\nSample batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Sample labels: {labels[:8].tolist()}")
    
    # Test the simple loader
    print("\n2. Testing simple CIFAR-10 loader:")
    simple_train, simple_test = load_cifar10_simple(batch_size=4)
    print(f"Simple train batches: {len(simple_train)}")
    print(f"Simple test batches: {len(simple_test)}")

"""
CIFAR-10 Data Loader

Provides data loading functionality for CIFAR-10 dataset with train/validation split.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def load_cifar10_data(batch_size=128, validation_split=0.1, data_augmentation=True):
    """
    Load CIFAR-10 data with train/validation split.
    
    Args:
        batch_size (int): Batch size for data loaders
        validation_split (float): Fraction of training data to use for validation
        data_augmentation (bool): Whether to apply data augmentation to training set
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Define transforms
    if data_augmentation:
        # Training transform with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # Simple training transform without augmentation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load the training data
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load the test data
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create train/validation split
    num_train = len(full_trainset)
    num_val = int(num_train * validation_split)
    num_train_final = num_train - num_val
    
    train_dataset, val_dataset = random_split(
        full_trainset, 
        [num_train_final, num_val],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
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
        testset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_classes():
    """Return CIFAR-10 class names."""
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_dataset_info():
    """Return CIFAR-10 dataset information."""
    return {
        'num_classes': 10,
        'input_channels': 3,
        'image_size': (32, 32),
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'classes': get_cifar10_classes()
    }


if __name__ == "__main__":
    # Test the data loader
    print("Testing CIFAR-10 data loader...")
    
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print(f"Sample batch shape: {images.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print(f"Image data type: {images.dtype}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Print class distribution in the sample
    classes = get_cifar10_classes()
    for i, class_name in enumerate(classes):
        count = (labels == i).sum().item()
        if count > 0:
            print(f"Class {i} ({class_name}): {count} samples")
    
    print("CIFAR-10 data loader test completed successfully!")

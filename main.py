import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
from load_mnist import get_mnist_dataset_local

def visualize_mnist_grid(images, labels, grid_size=6):
    """Display a grid of MNIST images with their labels"""
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle('MNIST Dataset Sample', fontsize=14)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(images):
                axes[i, j].imshow(images[idx], cmap='gray')
                axes[i, j].set_title(f'{labels[idx]}', fontsize=10)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

class MNISTDataset(Dataset):
    """Custom MNIST Dataset with augmentations using torchvision transforms"""
    
    def __init__(self, images, labels, transform=None):
        # Convert numpy arrays to torch tensors and add channel dimension
        self.images = torch.from_numpy(images).float().unsqueeze(1)  # Add channel dim: (N, 1, 28, 28)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(batch_size=64, apply_augmentation=True):
    """Create training and test data loaders with optional augmentation using torch transforms"""
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = get_mnist_dataset_local()
    
    # Create transform pipeline
    transform = None
    if apply_augmentation:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=180, fill=0),  # Rotate up to 180 degrees in each direction
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),  # Translate up to 10% of image size
        ])
    
    # Create datasets
    train_dataset = MNISTDataset(train_images, train_labels, transform=transform)
    test_dataset = MNISTDataset(test_images, test_labels, transform=None)  # No augmentation for test
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Load MNIST data using the imported function
    (train_images, train_labels), (test_images, test_labels) = get_mnist_dataset_local()
    print(f"Train samples: {len(train_images)} | Test samples: {len(test_images)}")
    
    # Create data loaders with augmentation
    train_loader, test_loader = create_data_loaders(batch_size=32, apply_augmentation=True)
    
    # Test the data loader
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
    
    # Visualize original images
    print("Original images:")
    visualize_mnist_grid(train_images[:36], train_labels[:36], grid_size=6)
    
    # Visualize augmented images
    print("Augmented images:")
    augmented_images = batch_images[:36].squeeze(1).numpy()  # Remove channel dim for visualization
    augmented_labels = batch_labels[:36].numpy()
    visualize_mnist_grid(augmented_images, augmented_labels, grid_size=6)
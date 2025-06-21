import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from load_mnist import get_mnist_dataset_local
from ebm import EnergyBasedModel

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

def add_noise_perturbations(images, noise_scale=0.1, noise_type='gaussian'):
    """
    Add noise perturbations to images to create negative samples
    Args:
        images: Input tensor of shape (batch_size, channels, height, width)
        noise_scale: Standard deviation of noise
        noise_type: Type of noise ('gaussian', 'uniform')
    Returns:
        Perturbed images
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(images) * noise_scale
    elif noise_type == 'uniform':
        noise = (torch.rand_like(images) - 0.5) * 2 * noise_scale
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'")
    
    # Add noise and clamp to valid pixel range [0, 1]
    perturbed_images = torch.clamp(images + noise, 0.0, 1.0)
    return perturbed_images

def energy_discrepancy_loss(positive_energy, negative_energy, margin=1.0):
    """
    Energy discrepancy loss for EBM training
    Args:
        positive_energy: Energy values for real/positive samples
        negative_energy: Energy values for negative/corrupted samples
        margin: Margin for the loss
    Returns:
        Loss value
    """
    # We want positive samples to have lower energy than negative samples
    # Loss = max(0, margin - (negative_energy - positive_energy))
    energy_diff = negative_energy - positive_energy
    loss = F.relu(margin - energy_diff).mean()
    return loss

def train_ebm(model, train_loader, num_epochs=10, lr=0.001, noise_scale_start=0.05, noise_scale_end=0.2, margin=1.0, device='cpu'):
    """
    Train the Energy-Based Model using energy discrepancy loss with noise scheduling
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(train_loader) * num_epochs
    
    model.train()
    batch_count = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Calculate current noise scale using linear scheduling
            progress = batch_count / total_batches
            current_noise_scale = noise_scale_start + (noise_scale_end - noise_scale_start) * progress
            
            # Generate negative samples with scheduled noise
            negative_images = add_noise_perturbations(images, noise_scale=current_noise_scale)
            
            # Get energy values for positive and negative samples
            positive_energy = model(images)
            negative_energy = model(negative_images)
            
            # Compute energy discrepancy loss
            loss = energy_discrepancy_loss(positive_energy, negative_energy, margin=margin)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            total_pos_energy += positive_energy.mean().item()
            total_neg_energy += negative_energy.mean().item()
            
            batch_count += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Pos Energy: {positive_energy.mean().item():.4f}, '
                      f'Neg Energy: {negative_energy.mean().item():.4f}, '
                      f'Noise Scale: {current_noise_scale:.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_energy = total_pos_energy / len(train_loader)
        avg_neg_energy = total_neg_energy / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Positive Energy: {avg_pos_energy:.4f}')
        print(f'  Average Negative Energy: {avg_neg_energy:.4f}')
        print(f'  Energy Gap: {avg_neg_energy - avg_pos_energy:.4f}')
        print(f'  Final Noise Scale: {current_noise_scale:.4f}')
        print('-' * 50)

def visualize_energy_comparison(model, test_loader, device='cpu', num_samples=36):
    """
    Visualize original images vs noisy images and their energy values
    """
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        images, labels = next(iter(test_loader))
        images = images.to(device)
        
        # Take first num_samples
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Generate negative samples
        negative_images = add_noise_perturbations(images, noise_scale=0.1)
        
        # Get energy values
        pos_energies = model(images).cpu().numpy().flatten()
        neg_energies = model(negative_images).cpu().numpy().flatten()
        
        # Convert to numpy for visualization
        orig_imgs = images.cpu().squeeze(1).numpy()
        noisy_imgs = negative_images.cpu().squeeze(1).numpy()
        
        # Create visualization
        fig, axes = plt.subplots(6, 6, figsize=(12, 12))
        fig.suptitle('Energy Comparison: Original vs Noisy Images', fontsize=16)
        
        for i in range(36):
            row = i // 6
            col = i % 6
            
            if i < len(orig_imgs):
                if i % 2 == 0:  # Show original image
                    idx = i // 2
                    axes[row, col].imshow(orig_imgs[idx], cmap='gray')
                    axes[row, col].set_title(f'Orig\nE:{pos_energies[idx]:.3f}', fontsize=8)
                else:  # Show noisy image
                    idx = i // 2
                    axes[row, col].imshow(noisy_imgs[idx], cmap='gray')
                    axes[row, col].set_title(f'Noisy\nE:{neg_energies[idx]:.3f}', fontsize=8)
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

def load_trained_ebm(model_path="trained_ebm.pth", device='cpu'):
    """
    Load a trained EBM model from file
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
    Returns:
        Loaded model and training info
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    model = EnergyBasedModel(
        input_channels=model_config['input_channels'],
        hidden_dim=model_config['hidden_dim']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Training info: {checkpoint['training_info']}")
    
    return model, checkpoint['training_info']

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = get_mnist_dataset_local()
    print(f"Train samples: {len(train_images)} | Test samples: {len(test_images)}")
    
    # Create data loaders with augmentation
    train_loader, test_loader = create_data_loaders(batch_size=64, apply_augmentation=True)
    
    # Initialize EBM model
    model = EnergyBasedModel(input_channels=1, hidden_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training parameters
    num_epochs = 10
    lr = 0.001
    noise_scale_start = 1.2   # Start with high noise (hard negatives)
    noise_scale_end = 0.01    # End with low noise (easier negatives)
    margin = 1.5
    
    # Train the model with noise scheduling
    print("Starting EBM training with noise scheduling...")
    train_ebm(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=lr,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        margin=margin,
        device=device
    )
    
    print("Training completed!")
    
    # Save the trained model
    model_save_path = "trained_ebm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_channels': 1,
            'hidden_dim': 128
        },
        'training_params': {
            'num_epochs': num_epochs,
            'lr': lr,
            'noise_scale_start': noise_scale_start,
            'noise_scale_end': noise_scale_end,
            'margin': margin
        }
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Visualize results
    print("Visualizing energy comparison...")
    visualize_energy_comparison(model, test_loader, device=device, num_samples=18)
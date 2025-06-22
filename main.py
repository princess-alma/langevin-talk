import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from load_mnist import get_mnist_dataset_local
from ebm import EnergyBasedModel, load_trained_ebm
from sampling import LangevinSampler

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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    total_batches = len(train_loader) * num_epochs
    
    model.train()
    batch_count = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                   leave=True, unit='batch')
        
        for batch_idx, (images, labels) in enumerate(pbar):
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
            pos_energy_mean = positive_energy.mean().item()
            neg_energy_mean = negative_energy.mean().item()
            total_pos_energy += pos_energy_mean
            total_neg_energy += neg_energy_mean
            
            batch_count += 1
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pos_E': f'{pos_energy_mean:.3f}',
                'Neg_E': f'{neg_energy_mean:.3f}',
                'Noise': f'{current_noise_scale:.3f}'
            })
        
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

def train_ebm_random_noise(model, train_loader, num_epochs=10, lr=0.001, noise_levels=None, margin=1.0, device='cpu'):
    """
    Train the Energy-Based Model using energy discrepancy loss with random noise sampling
    """
    if noise_levels is None:
        noise_levels = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.01]
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        noise_usage_count = {level: 0 for level in noise_levels}
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Random Noise)', 
                   leave=True, unit='batch')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Randomly sample noise scale from predefined levels
            current_noise_scale = random.choice(noise_levels)
            noise_usage_count[current_noise_scale] += 1
            
            # Generate negative samples with randomly sampled noise
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
            pos_energy_mean = positive_energy.mean().item()
            neg_energy_mean = negative_energy.mean().item()
            total_pos_energy += pos_energy_mean
            total_neg_energy += neg_energy_mean
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pos_E': f'{pos_energy_mean:.3f}',
                'Neg_E': f'{neg_energy_mean:.3f}',
                'Noise': f'{current_noise_scale:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_energy = total_pos_energy / len(train_loader)
        avg_neg_energy = total_neg_energy / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Positive Energy: {avg_pos_energy:.4f}')
        print(f'  Average Negative Energy: {avg_neg_energy:.4f}')
        print(f'  Energy Gap: {avg_neg_energy - avg_pos_energy:.4f}')
        
        # Show noise level usage statistics
        total_batches_epoch = len(train_loader)
        print(f'  Noise level usage this epoch:')
        for level in sorted(noise_levels, reverse=True):
            count = noise_usage_count[level]
            percentage = (count / total_batches_epoch) * 100
            print(f'    {level:.2f}: {count} batches ({percentage:.1f}%)')
        print('-' * 50)

def train_ebm_with_langevin(model, train_loader, num_epochs=10, lr=0.001, margin=1.0, 
                           langevin_steps=20, langevin_step_size=0.1, langevin_noise_scale=0.01, 
                           device='cpu'):
    """
    Train the Energy-Based Model using Langevin sampling for negative sample generation
    Args:
        model: The EBM model to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate for model parameters
        margin: Margin for energy discrepancy loss
        langevin_steps: Number of Langevin MCMC steps for negative sampling
        langevin_step_size: Step size for Langevin updates
        langevin_noise_scale: Noise scale for Langevin sampling
        device: Device to run training on
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Initialize Langevin sampler with maximize_energy=True for negative sample generation
    sampler = LangevinSampler(
        model=model,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps,
        device=device,
        maximize_energy=True  # Use gradient ascent to maximize energy
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        total_energy_diff = 0.0
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Langevin)', 
                   leave=True, unit='batch')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate negative samples using Langevin sampling (gradient ascent)
            negative_images, _ = sampler.sample_negatives(images)
            
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
            pos_energy_mean = positive_energy.mean().item()
            neg_energy_mean = negative_energy.mean().item()
            total_pos_energy += pos_energy_mean
            total_neg_energy += neg_energy_mean
            total_energy_diff += (neg_energy_mean - pos_energy_mean)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pos_E': f'{pos_energy_mean:.3f}',
                'Neg_E': f'{neg_energy_mean:.3f}',
                'ΔE': f'{neg_energy_mean - pos_energy_mean:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_energy = total_pos_energy / len(train_loader)
        avg_neg_energy = total_neg_energy / len(train_loader)
        avg_energy_diff = total_energy_diff / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Positive Energy: {avg_pos_energy:.4f}')
        print(f'  Average Negative Energy: {avg_neg_energy:.4f}')
        print(f'  Average Energy Difference: {avg_energy_diff:.4f}')
        print(f'  Langevin Steps: {langevin_steps}')
        print(f'  Langevin Step Size: {langevin_step_size}')
        print('-' * 50)

def visualize_langevin_negatives(model, test_loader, langevin_steps=20, langevin_step_size=0.1, 
                                langevin_noise_scale=0.01, device='cpu', num_samples=9):
    """
    Visualize the negative samples generated using Langevin sampling
    """
    model.eval()
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    
    # Initialize Langevin sampler for maximizing energy
    sampler = LangevinSampler(
        model=model,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps,
        device=device,
        maximize_energy=True
    )
    
    # Generate negative samples using the improved sampler
    negative_images, _ = sampler.sample_negatives(images)
    
    # Get energy values
    with torch.no_grad():
        pos_energies = model(images).cpu().numpy().flatten()
        neg_energies = model(negative_images).cpu().numpy().flatten()
    
    # Convert to numpy for visualization
    orig_imgs = images.cpu().squeeze(1).numpy()
    neg_imgs = negative_images.cpu().squeeze(1).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    fig.suptitle('Langevin Negative Sample Generation', fontsize=16)
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(orig_imgs[i], cmap='gray')
        axes[0, i].set_title(f'Orig\nE:{pos_energies[i]:.3f}', fontsize=8)
        axes[0, i].axis('off')
        
        # Negative image
        axes[1, i].imshow(neg_imgs[i], cmap='gray')
        axes[1, i].set_title(f'Neg\nE:{neg_energies[i]:.3f}', fontsize=8)
        axes[1, i].axis('off')
        
        # Difference
        diff = neg_imgs[i] - orig_imgs[i]
        axes[2, i].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        axes[2, i].set_title(f'Diff\nΔE:{neg_energies[i]-pos_energies[i]:.3f}', fontsize=8)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

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

def train_ebm_from_noise(model, train_loader, num_epochs=10, lr=0.001, margin=1.0, 
                         langevin_steps=20, langevin_step_size=0.1, langevin_noise_scale=0.01, 
                         device='cpu'):
    """
    Train the EBM by generating negative samples starting from random noise.
    This method finds spurious low-energy regions ("holes") in the energy landscape
    and trains the model to raise their energy.

    - Positives: Real data samples from the dataset.
    - Negatives: Samples generated by minimizing energy (gradient descent) starting from Gaussian noise.
    
    Args:
        model: The EBM model to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate for model parameters
        margin: Margin for energy discrepancy loss
        langevin_steps: Number of Langevin MCMC steps for negative sampling
        langevin_step_size: Step size for Langevin updates
        langevin_noise_scale: Noise scale for Langevin sampling
        device: Device to run training on
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Initialize Langevin sampler for minimizing energy from noise
    minimizer_sampler = LangevinSampler(
        model=model,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps,
        device=device,
        maximize_energy=False  # Gradient descent from random noise
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        total_energy_diff = 0.0
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (From Noise)', 
                   leave=True, unit='batch')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate negative samples: Minimize energy from Gaussian noise
            # Use uniform noise for better initialization
            random_noise = torch.rand_like(images)
            negative_images = minimizer_sampler.sample(random_noise)
            
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
            pos_energy_mean = positive_energy.mean().item()
            neg_energy_mean = negative_energy.mean().item()
            
            total_pos_energy += pos_energy_mean
            total_neg_energy += neg_energy_mean
            total_energy_diff += (neg_energy_mean - pos_energy_mean)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pos_E': f'{pos_energy_mean:.3f}',
                'Neg_E': f'{neg_energy_mean:.3f}',
                'ΔE': f'{neg_energy_mean - pos_energy_mean:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_energy = total_pos_energy / len(train_loader)
        avg_neg_energy = total_neg_energy / len(train_loader)
        avg_energy_diff = total_energy_diff / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Positive Energy: {avg_pos_energy:.4f}')
        print(f'  Average Negative Energy: {avg_neg_energy:.4f}')
        print(f'  Average Energy Difference: {avg_energy_diff:.4f}')
        print(f'  Langevin Steps: {langevin_steps}')
        print(f'  Langevin Step Size: {langevin_step_size}')
        print('-' * 50)

def visualize_hybrid_negatives(model, test_loader, langevin_steps=20, langevin_step_size=0.1, 
                              langevin_noise_scale=0.01, device='cpu', num_samples=6):
    """
    Visualize the hybrid negative samples: from positive images and from Gaussian noise
    """
    model.eval()
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    
    # Initialize both samplers
    maximizer_sampler = LangevinSampler(
        model=model,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps,
        device=device,
        maximize_energy=True
    )
    
    minimizer_sampler = LangevinSampler(
        model=model,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps,
        device=device,
        maximize_energy=False
    )
    
    # Generate negative samples from positive images (maximize energy)
    neg_from_pos, _ = maximizer_sampler.sample_negatives(images)
    
    # Generate negative samples from random noise (minimize energy)
    random_noise = torch.randn_like(images)
    random_noise = torch.clamp(random_noise, 0.0, 1.0)
    neg_from_noise, _ = minimizer_sampler.sample_with_trajectory(random_noise, save_steps=2, maximize_energy=False)
    
    # Get energy values
    with torch.no_grad():
        pos_energies = model(images).cpu().numpy().flatten()
        neg_pos_energies = model(neg_from_pos).cpu().numpy().flatten()
        neg_noise_energies = model(neg_from_noise).cpu().numpy().flatten()
        noise_energies = model(random_noise).cpu().numpy().flatten()
    
    # Convert to numpy for visualization
    orig_imgs = images.cpu().squeeze(1).numpy()
    neg_pos_imgs = neg_from_pos.cpu().squeeze(1).numpy()
    neg_noise_imgs = neg_from_noise.cpu().squeeze(1).numpy()
    noise_imgs = random_noise.cpu().squeeze(1).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
    fig.suptitle('Hybrid Negative Sample Generation', fontsize=16)
    
    for i in range(num_samples):
        # Original positive image
        axes[0, i].imshow(orig_imgs[i], cmap='gray')
        axes[0, i].set_title(f'Positive\nE:{pos_energies[i]:.3f}', fontsize=10)
        axes[0, i].axis('off')
        
        # Negative from positive (maximized)
        axes[1, i].imshow(neg_pos_imgs[i], cmap='gray')
        axes[1, i].set_title(f'Neg (Max)\nE:{neg_pos_energies[i]:.3f}', fontsize=10)
        axes[1, i].axis('off')
        
        # Initial random noise
        axes[2, i].imshow(noise_imgs[i], cmap='gray')
        axes[2, i].set_title(f'Init Noise\nE:{noise_energies[i]:.3f}', fontsize=10)
        axes[2, i].axis('off')
        
        # Negative from noise (minimized)
        axes[3, i].imshow(neg_noise_imgs[i], cmap='gray')
        axes[3, i].set_title(f'Neg (Min)\nE:{neg_noise_energies[i]:.3f}', fontsize=10)
        axes[3, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Positive Samples', rotation=90, labelpad=20)
    axes[1, 0].set_ylabel('Negatives from Pos', rotation=90, labelpad=20)
    axes[2, 0].set_ylabel('Initial Noise', rotation=90, labelpad=20)
    axes[3, 0].set_ylabel('Negatives from Noise', rotation=90, labelpad=20)
    
    plt.tight_layout()
    plt.show()

def train_ebm_with_replay_buffer(model, train_loader, num_epochs=10, lr=0.0001, margin=10.0,
                                 langevin_steps=60, langevin_step_size=0.1, langevin_noise_scale=0.01,
                                 buffer_size=10000, device='cpu'):
    """
    Train the EBM using a replay buffer for persistent contrastive divergence.
    This is the standard and most effective way to train EBMs.
    
    The replay buffer stores negative samples from previous iterations, giving the
    Langevin sampler a "head start" to find more realistic and challenging negatives.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initialize the replay buffer with random noise
    replay_buffer = torch.rand(buffer_size, 1, 28, 28).to(device)

    # Sampler for MINIMIZING energy
    sampler = LangevinSampler(
        model=model, 
        step_size=langevin_step_size, 
        noise_scale=langevin_noise_scale,
        num_steps=langevin_steps, 
        device=device, 
        maximize_energy=False  # Minimize energy to find low-energy regions
    )

    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_energy = 0.0
        total_neg_energy = 0.0
        total_energy_diff = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Replay Buffer)', 
                   leave=True, unit='batch')

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)

            # --- Sample from Replay Buffer ---
            # 1. Get random indices for samples from the buffer
            buffer_indices = torch.randint(0, buffer_size, (batch_size,))
            initial_neg_samples = replay_buffer[buffer_indices].clone()

            # 2. Run Langevin dynamics starting from the buffered samples to get new negatives
            # We detach to stop gradients from flowing back into the buffer from previous steps
            negative_images = sampler.sample(initial_neg_samples.detach())

            # --- Compute Loss ---
            positive_energy = model(images)
            negative_energy = model(negative_images)

            loss = energy_discrepancy_loss(positive_energy, negative_energy, margin=margin)

            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Update Replay Buffer ---
            # 3. Replace the old samples in the buffer with the newly generated ones
            replay_buffer[buffer_indices] = negative_images.detach()

            # Track statistics
            total_loss += loss.item()
            pos_energy_mean = positive_energy.mean().item()
            neg_energy_mean = negative_energy.mean().item()
            
            total_pos_energy += pos_energy_mean
            total_neg_energy += neg_energy_mean
            total_energy_diff += (neg_energy_mean - pos_energy_mean)

            # --- Logging ---
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pos_E': f'{pos_energy_mean:.3f}',
                'Neg_E': f'{neg_energy_mean:.3f}',
                'ΔE': f'{neg_energy_mean - pos_energy_mean:.3f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_pos_energy = total_pos_energy / len(train_loader)
        avg_neg_energy = total_neg_energy / len(train_loader)
        avg_energy_diff = total_energy_diff / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Positive Energy: {avg_pos_energy:.4f}')
        print(f'  Average Negative Energy: {avg_neg_energy:.4f}')
        print(f'  Average Energy Difference: {avg_energy_diff:.4f}')
        print(f'  Margin: {margin} | Langevin Steps: {langevin_steps}')
        print('-' * 50)

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
    
    # --- Tuned Hyperparameters for Replay Buffer Training ---
    num_epochs = 5  # Train for longer
    lr = 1e-4        # Lower learning rate for more stable learning
    margin = 10.0    # Increased margin to force larger energy gaps
    langevin_steps = 60  # More steps for better negative samples
    langevin_step_size = 0.1
    langevin_noise_scale = 0.01

    print("Starting EBM training with Replay Buffer...")
    train_ebm_with_replay_buffer(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=lr,
        margin=margin,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
        langevin_noise_scale=langevin_noise_scale,
        device=device
    )

    print("Training completed!")
    
    # Save the model with correct parameters for replay buffer training
    model_save_path = "trained_ebm_replay.pth"
    training_params = {
        'method': 'train_ebm_with_replay_buffer',
        'num_epochs': num_epochs,
        'lr': lr,
        'margin': margin,
        'langevin_steps': langevin_steps,
        'langevin_step_size': langevin_step_size,
        'langevin_noise_scale': langevin_noise_scale,
        'buffer_size': 10000
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': { 'input_channels': 1, 'hidden_dim': 128 },
        'training_info': training_params
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Training Info: {training_params}")

    # Visualization functions are still available for separate use if needed
    # visualize_energy_comparison(model, test_loader, device=device, num_samples=18)
    # visualize_langevin_negatives(model, test_loader, langevin_steps=20, langevin_step_size=0.1, langevin_noise_scale=0.01, device=device, num_samples=9)
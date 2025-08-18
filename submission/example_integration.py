"""
Example: Using SGLD Optimizer with Energy-Based Models

This example demonstrates how to use the SGLD optimizer for training energy-based models,
integrating with the existing codebase in the langevin-talk repository.
"""

import sys
import os
sys.path.append('..')  # Add parent directory to path

import torch
import torch.nn as nn
import numpy as np
from sgld import SGLD, polynomial_schedule, cosine_annealing_schedule

# Try to import the EBM from parent directory
try:
    from ebm import SimpleEBM
    from load_mnist import get_mnist_dataset_local, MNISTDataset
    from torch.utils.data import DataLoader
    HAVE_EBM = True
except ImportError:
    print("Could not import EBM modules - creating simple EBM for demonstration")
    HAVE_EBM = False
    
    class SimpleEBM(nn.Module):
        """Simple EBM for demonstration when main modules aren't available."""
        def __init__(self, input_channels=1, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze()


def train_ebm_with_sgld(model, train_loader, num_epochs=5, device='cpu'):
    """
    Train an Energy-Based Model using the SGLD optimizer.
    
    This function demonstrates how to replace standard optimizers (Adam, SGD) 
    with SGLD for Bayesian learning.
    """
    model.to(device)
    
    # Initialize SGLD optimizer with appropriate parameters
    optimizer = SGLD(
        model.parameters(),
        lr=0.01,                    # Base learning rate
        temperature=0.1,            # Temperature for Langevin noise
        momentum=0.9,               # Optional momentum
        weight_decay=1e-4,          # L2 regularization
        lr_decay=0.55,              # Polynomial decay rate
        min_lr=1e-6,                # Minimum learning rate
        noise_decay=0.99            # Gradual noise reduction
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Simple contrastive loss for demonstration
            # In practice, you'd use proper EBM training with negative sampling
            optimizer.zero_grad()
            
            # Positive samples (real data)
            pos_energy = model(data)
            
            # Negative samples (random noise for simplicity)
            neg_data = torch.randn_like(data)
            neg_energy = model(neg_data)
            
            # Contrastive loss: minimize positive energy, maximize negative energy
            loss = pos_energy.mean() - neg_energy.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                info = optimizer.get_info()
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: '
                      f'Loss = {loss.item():.4f}, '
                      f'LR = {info["current_lr"]:.6f}, '
                      f'Temp = {info["current_temperature"]:.6f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}')
        print('-' * 60)


def demonstrate_schedule_comparison():
    """
    Demonstrate different learning rate schedules with SGLD.
    """
    print("=== Comparing Learning Rate Schedules ===")
    
    # Create a simple 2D optimization problem
    def optimize_with_schedule(schedule_name, lr_schedule=None, temp_schedule=None):
        # Simple quadratic function
        x = torch.tensor([2.0, -1.5], requires_grad=True)
        
        if lr_schedule is None:
            # Standard SGLD with polynomial decay
            optimizer = SGLD([x], lr=0.02, temperature=0.05)
        else:
            # Custom schedule
            from sgld import SGLDWithCustomSchedule
            optimizer = SGLDWithCustomSchedule(
                [x], lr=0.02, temperature=0.05,
                lr_schedule=lr_schedule, temperature_schedule=temp_schedule
            )
        
        losses = []
        for step in range(100):
            optimizer.zero_grad()
            loss = (x[0] - 0)**2 + (x[1] - 0)**2  # Target: (0, 0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_pos = x.detach().numpy()
        final_loss = losses[-1]
        print(f"{schedule_name}: Final position = [{final_pos[0]:.4f}, {final_pos[1]:.4f}], "
              f"Final loss = {final_loss:.6f}")
        
        return losses
    
    # Test different schedules
    standard_losses = optimize_with_schedule("Standard Polynomial")
    
    poly_losses = optimize_with_schedule(
        "Custom Polynomial", 
        lr_schedule=polynomial_schedule(0.05, decay_rate=0.8)
    )
    
    cosine_losses = optimize_with_schedule(
        "Cosine Annealing",
        lr_schedule=cosine_annealing_schedule(0.05, min_lr=0.001, period=50)
    )
    
    print()


def demonstrate_temperature_annealing():
    """
    Demonstrate temperature annealing for exploration-to-exploitation transition.
    """
    print("=== Temperature Annealing for Exploration-Exploitation ===")
    
    # Multi-modal optimization problem
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    
    # Custom temperature schedule: start high for exploration, end low for exploitation
    def temperature_schedule(step):
        # Start with high temperature, decay to low temperature
        return 0.5 * (0.95 ** step) + 0.001
    
    from sgld import SGLDWithCustomSchedule
    optimizer = SGLDWithCustomSchedule(
        [x], 
        lr=0.01, 
        temperature=0.5,
        temperature_schedule=temperature_schedule
    )
    
    positions = []
    temperatures = []
    
    for step in range(200):
        optimizer.zero_grad()
        
        # Multi-modal function with two peaks at (-1,-1) and (1,1)
        dist1 = (x[0] + 1)**2 + (x[1] + 1)**2
        dist2 = (x[0] - 1)**2 + (x[1] - 1)**2
        loss = -torch.exp(-dist1) - torch.exp(-dist2)  # Negative for maximization
        
        loss.backward()
        optimizer.step()
        
        positions.append(x.detach().clone().numpy())
        temperatures.append(temperature_schedule(step))
        
        if step % 50 == 0:
            current_temp = temperature_schedule(step)
            print(f"Step {step}: Position = [{x[0].item():.4f}, {x[1].item():.4f}], "
                  f"Temperature = {current_temp:.6f}, Loss = {loss.item():.6f}")
    
    final_pos = x.detach().numpy()
    print(f"Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}]")
    print("Note: High initial temperature allows exploration between modes, "
          "low final temperature enables precise convergence")
    print()


def main():
    """Main demonstration function."""
    print("SGLD Optimizer Integration Examples")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Example 1: Basic EBM training with SGLD
    print("=== Example 1: EBM Training with SGLD ===")
    
    # Create a simple EBM
    model = SimpleEBM(input_channels=1, hidden_dim=64)
    
    # Create dummy data loader if real MNIST isn't available
    if HAVE_EBM:
        try:
            train_images, _ = get_mnist_dataset_local()
            train_dataset = MNISTDataset(train_images[:1000], torch.zeros(1000))  # Subset
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        except:
            # Create dummy data
            dummy_data = torch.randn(100, 1, 28, 28)
            dummy_labels = torch.zeros(100)
            train_loader = DataLoader(list(zip(dummy_data, dummy_labels)), batch_size=32)
    else:
        # Create dummy data
        dummy_data = torch.randn(100, 1, 28, 28)
        dummy_labels = torch.zeros(100)
        train_loader = DataLoader(list(zip(dummy_data, dummy_labels)), batch_size=32)
    
    # Train with SGLD
    train_ebm_with_sgld(model, train_loader, num_epochs=2, device=device)
    
    # Example 2: Learning rate schedule comparison
    demonstrate_schedule_comparison()
    
    # Example 3: Temperature annealing
    demonstrate_temperature_annealing()
    
    print("Integration examples completed!")
    print("\nKey Integration Points:")
    print("✓ Replace standard optimizers (Adam, SGD) with SGLD")
    print("✓ Use temperature parameter for exploration control")
    print("✓ Apply learning rate decay for convergence guarantees")
    print("✓ Custom schedules for advanced use cases")
    print("✓ Monitor optimizer state for debugging/analysis")


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
import numpy as np
from load_mnist import get_mnist_dataset_local

(train_images, train_labels), (test_images, test_labels) = get_mnist_dataset_local()

class LangevinSampler:
    """Langevin Dynamics Sampler for Energy-Based Models"""

    def __init__(self, model, step_size=0.1, noise_scale=0.01, num_steps=100, device='cpu', maximize_energy=False):
        """
        Initialize the Langevin sampler.
        Args:
            model: The trained energy-based model.
            step_size: Step size for gradient updates.
            noise_scale: Scale of Gaussian noise added at each step.
            num_steps: Number of Langevin steps to perform.
            device: Device to run the sampling on.
            maximize_energy: If True, perform gradient ascent to maximize energy (for negative sample generation).
                           If False, perform gradient descent to minimize energy (for sampling from the model).
        """
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.num_steps = num_steps
        self.device = device
        self.maximize_energy = maximize_energy

    def sample_with_trajectory(self, initial_samples, save_steps=36, maximize_energy=None):
        """
        Perform Langevin sampling and save intermediate steps.
        Args:
            initial_samples: Tensor of initial samples (batch_size, channels, height, width).
            save_steps: Number of steps to save (including first and last).
            maximize_energy: Override the instance setting for this specific call.
        Returns:
            Final samples and trajectory of saved steps.
        """
        # Use override if provided, otherwise use instance setting
        if maximize_energy is None:
            maximize_energy = self.maximize_energy
            
        # --- FIX: Use model.eval() for stable BatchNorm statistics ---
        # BatchNorm should use running statistics, not mini-batch statistics during sampling
        self.model.eval()
        
        samples = initial_samples.clone().to(self.device)
        samples.requires_grad_(True)
        
        # Calculate which steps to save
        if save_steps <= 2:
            save_indices = {0, self.num_steps - 1}
        else:
            save_indices = set(np.linspace(0, self.num_steps - 1, save_steps, dtype=int))
        
        trajectory = []
        
        for step in range(self.num_steps):
            # Save current state if it's a save step
            if step in save_indices:
                trajectory.append(samples.detach().clone().cpu())
            
            energy = self.model.energy(samples).sum()
            
            # Use torch.autograd.grad to get gradients w.r.t. samples ONLY
            # This prevents contamination of model parameter gradients
            grad_x = torch.autograd.grad(energy, samples, only_inputs=True)[0]

            # Langevin update is performed within a no_grad context
            with torch.no_grad():
                if maximize_energy:
                    # Gradient ascent: move in direction of gradient to maximize energy
                    samples.data += self.step_size * grad_x
                else:
                    # Gradient descent: move opposite to gradient to minimize energy
                    samples.data -= self.step_size * grad_x
                
                # Add noise
                samples.data += self.noise_scale * torch.randn_like(samples)
                
                # Clamp to valid pixel range
                samples.data = torch.clamp(samples.data, 0.0, 1.0)
        
        # Ensure the final step is always included in the trajectory
        if (self.num_steps - 1) not in save_indices or not trajectory:
             trajectory.append(samples.detach().clone().cpu())

        return samples.detach(), trajectory

    def sample(self, initial_samples, maximize_energy=None):
        """
        Perform Langevin sampling starting from initial samples.
        Args:
            initial_samples: Tensor of initial samples (batch_size, channels, height, width).
            maximize_energy: Override the instance setting for this specific call.
        Returns:
            Final samples after Langevin dynamics.
        """
        final_samples, _ = self.sample_with_trajectory(initial_samples, save_steps=2, maximize_energy=maximize_energy)
        return final_samples

    def sample_negatives(self, positive_samples, save_steps=2):
        """
        Generate negative samples by maximizing energy starting from positive samples.
        This is a convenience method specifically for training EBMs.
        Args:
            positive_samples: Tensor of positive samples to start from.
            save_steps: Number of trajectory steps to save.
        Returns:
            Final negative samples after gradient ascent.
        """
        return self.sample_with_trajectory(positive_samples, save_steps=save_steps, maximize_energy=True)

def visualize_batch_sampling(model, num_samples=36, num_steps=100, step_size=0.1, noise_scale=0.01, device='cpu', title="Batch Sampling Results"):
    """
    Visualize the final results of batch sampling by evolving 36 random samples.
    Shows only the final generated samples in a 6x6 grid.
    Args:
        model: The trained energy-based model.
        num_samples: Number of samples to generate (default 36 for 6x6 grid).
        num_steps: Number of Langevin steps to perform.
        step_size: Step size for gradient updates.
        noise_scale: Scale of Gaussian noise added at each step.
        device: Device to run the sampling on.
        title: Title for the plot.
    """
    import matplotlib.pyplot as plt
    
    # Initialize Langevin sampler for generation (minimizing energy)
    sampler = LangevinSampler(
        model=model,
        step_size=step_size,
        noise_scale=noise_scale,
        num_steps=num_steps,
        device=device,
        maximize_energy=False  # Minimize energy to generate samples
    )
    
    # Start with random noise
    initial_samples = torch.rand(num_samples, 1, 28, 28).to(device)
    
    print(f"Starting batch sampling with {num_samples} samples for {num_steps} steps...")
    
    # Perform Langevin sampling (only need final results)
    final_samples = sampler.sample(initial_samples)
    
    # Convert to numpy for visualization
    final_images = final_samples.cpu().squeeze(1).numpy()
    
    # Get energy values for the final samples
    with torch.no_grad():
        model.eval()
        final_energies = model(final_samples).cpu().numpy().flatten()
    
    # Create 6x6 grid visualization showing only final results
    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    fig.suptitle(f'{title}\n{num_steps} Langevin Steps (Step Size: {step_size}, Noise: {noise_scale})', fontsize=16)
    
    for i in range(6):
        for j in range(6):
            idx = i * 6 + j
            
            if idx < len(final_images):
                # Show final generated sample
                axes[i, j].imshow(final_images[idx], cmap='gray')
                axes[i, j].set_title(f'E: {final_energies[idx]:.2f}', fontsize=10)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    # Add energy statistics at the bottom
    avg_final_energy = np.mean(final_energies)
    min_energy = np.min(final_energies)
    max_energy = np.max(final_energies)
    
    plt.figtext(0.5, 0.02, f'Energy Stats - Avg: {avg_final_energy:.2f}, Min: {min_energy:.2f}, Max: {max_energy:.2f}', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for energy stats
    plt.show()
    
    print(f"Batch sampling completed!")
    print(f"Average final energy: {avg_final_energy:.4f}")
    print(f"Energy range: [{min_energy:.2f}, {max_energy:.2f}]")

def visualize_sampling_trajectory(trajectory, sample_idx=0, title="Langevin Sampling Trajectory"):
    """
    Visualize the sampling trajectory in a 6x6 grid.
    Args:
        trajectory: List of tensor samples at different steps.
        sample_idx: Index of the sample to visualize from each batch.
        title: Title for the plot.
    """
    import matplotlib.pyplot as plt
    
    # Take exactly 36 steps, padding with the last step if needed
    steps_to_show = 36
    if len(trajectory) < steps_to_show:
        # Pad with the last step
        trajectory = trajectory + [trajectory[-1]] * (steps_to_show - len(trajectory))
    elif len(trajectory) > steps_to_show:
        # Take the first 36 steps
        trajectory = trajectory[:steps_to_show]
    
    fig, axes = plt.subplots(6, 6, figsize=(8, 8))  # Much smaller: (8, 8)
    fig.suptitle(title, fontsize=12)  # Smaller title
    
    for i, step_samples in enumerate(trajectory):
        row = i // 6
        col = i % 6
        
        # Extract the specific sample and remove channel dimension
        image = step_samples[sample_idx, 0].numpy()
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'{i}', fontsize=6)  # Just step number, smaller font
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from ebm import load_trained_ebm
    from load_mnist import get_mnist_dataset_local
    import matplotlib.pyplot as plt

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model_path = "trained_ebm_replay.pth"
    model, training_info = load_trained_ebm(model_path, device=device)

    # Initialize Langevin sampler
    sampler = LangevinSampler(model, step_size=0.01, noise_scale=0.01, num_steps=100, device=device)

    # Select initial samples (random noise)
    batch_size = 1
    # Pick a random image from test_images
    random_idx = np.random.randint(0, len(train_images))
    initial_img = train_images[random_idx]
    initial_samples = torch.tensor(initial_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    noise = torch.randn_like(initial_samples) * 1  # Add some noise
    initial_samples = noise  # Add noise to the image
    print(f"Starting Langevin sampling with {batch_size} samples...")

    # Perform Langevin sampling with trajectory
    #final_samples, trajectory = sampler.sample_with_trajectory(initial_samples, save_steps=36)

    #print(f"Sampling completed! Saved {len(trajectory)} steps.")
    
    # Visualize trajectory for the first sample
    #visualize_sampling_trajectory(trajectory, sample_idx=0, title="Langevin Sampling: Random Noise → MNIST-like Digit")
    
    # Also show trajectories for other samples
    #for i in range(1, min(batch_size, 3)):
    #    visualize_sampling_trajectory(trajectory, sample_idx=i, title=f"Langevin Sampling Trajectory - Sample {i+1}")
    
    # Visualize batch sampling results
    visualize_batch_sampling(model, num_samples=36, num_steps=2000, step_size=0.1, noise_scale=0.01, device=device, title="Batch Sampling Results")
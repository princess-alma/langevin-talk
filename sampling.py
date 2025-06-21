import torch
import torch.nn.functional as F
import numpy as np

class LangevinSampler:
    """Langevin Dynamics Sampler for Energy-Based Models"""

    def __init__(self, model, step_size=0.1, noise_scale=0.01, num_steps=100, device='cpu'):
        """
        Initialize the Langevin sampler.
        Args:
            model: The trained energy-based model.
            step_size: Step size for gradient updates.
            noise_scale: Scale of Gaussian noise added at each step.
            num_steps: Number of Langevin steps to perform.
            device: Device to run the sampling on.
        """
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.num_steps = num_steps
        self.device = device

    def sample_with_trajectory(self, initial_samples, save_steps=36):
        """
        Perform Langevin sampling and save intermediate steps.
        Args:
            initial_samples: Tensor of initial samples (batch_size, channels, height, width).
            save_steps: Number of steps to save (including first and last).
        Returns:
            Final samples and trajectory of saved steps.
        """
        # Put model in train mode to enable gradients
        self.model.train()
        
        samples = initial_samples.clone().to(self.device)
        samples.requires_grad_(True)
        
        # Calculate which steps to save (first, last, and evenly spaced between)
        if save_steps <= 2:
            save_indices = [0, self.num_steps - 1]
        else:
            # First and last steps
            save_indices = [0, self.num_steps - 1]
            # Evenly spaced steps in between
            if save_steps > 2:
                middle_steps = np.linspace(1, self.num_steps - 2, save_steps - 2, dtype=int)
                save_indices = [0] + sorted(middle_steps.tolist()) + [self.num_steps - 1]
                # Remove duplicates and sort
                save_indices = sorted(list(set(save_indices)))
                # Take exactly save_steps
                if len(save_indices) > save_steps:
                    save_indices = save_indices[:save_steps]
        
        trajectory = []
        
        for step in range(self.num_steps):
            # Save current state if it's a save step
            if step in save_indices:
                trajectory.append(samples.detach().clone().cpu())
            
            # Compute energy gradient
            if samples.grad is not None:
                samples.grad.zero_()
                
            energy = self.model.energy(samples).sum()
            energy.backward()

            # Langevin update: gradient descent + noise
            with torch.no_grad():
                if samples.grad is not None:
                    samples.data -= self.step_size * samples.grad
                samples.data += self.noise_scale * torch.randn_like(samples)
                samples.data = torch.clamp(samples.data, 0.0, 1.0)  # Keep samples in valid range [0, 1]
        
        # Make sure we have the final step
        if (self.num_steps - 1) not in save_indices:
            trajectory.append(samples.detach().clone().cpu())

        # Put model back in eval mode
        self.model.eval()
        
        return samples.detach(), trajectory

    def sample(self, initial_samples):
        """
        Perform Langevin sampling starting from initial samples.
        Args:
            initial_samples: Tensor of initial samples (batch_size, channels, height, width).
        Returns:
            Final samples after Langevin dynamics.
        """
        final_samples, _ = self.sample_with_trajectory(initial_samples, save_steps=2)
        return final_samples

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
    model_path = "trained_ebm.pth"
    model, training_info = load_trained_ebm(model_path, device=device)

    # Initialize Langevin sampler
    sampler = LangevinSampler(model, step_size=0.1, noise_scale=0.01, num_steps=1000, device=device)

    # Select initial samples (random noise)
    batch_size = 1
    initial_samples = torch.rand(batch_size, 1, 28, 28)  # Random noise in [0, 1]
    print(f"Starting Langevin sampling with {batch_size} samples...")

    # Perform Langevin sampling with trajectory
    final_samples, trajectory = sampler.sample_with_trajectory(initial_samples, save_steps=36)

    print(f"Sampling completed! Saved {len(trajectory)} steps.")
    
    # Visualize trajectory for the first sample
    visualize_sampling_trajectory(trajectory, sample_idx=0, title="Langevin Sampling: Random Noise → MNIST-like Digit")
    
    # Also show trajectories for other samples
    #for i in range(1, min(batch_size, 3)):
    #    visualize_sampling_trajectory(trajectory, sample_idx=i, title=f"Langevin Sampling Trajectory - Sample {i+1}")
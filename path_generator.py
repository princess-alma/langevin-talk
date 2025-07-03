import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Optional
from path_visualization import PathVisualization, generate_sample_distribution


class PathGenerator:
    """
    Utility class for generating SGD and SGLD optimization paths.
    
    This class provides methods to run SGD and SGLD on various objective functions
    and return the paths in a format suitable for the PathVisualization class.
    """
    
    def __init__(self, 
                 objective_func: Callable[[torch.Tensor], torch.Tensor],
                 x_range: Tuple[float, float] = (-3, 3),
                 y_range: Tuple[float, float] = (-3, 3)):
        """
        Initialize the path generator.
        
        Args:
            objective_func: Function that takes a 2D tensor and returns scalar loss
            x_range: Range for x coordinate
            y_range: Range for y coordinate
        """
        self.objective_func = objective_func
        self.x_range = x_range
        self.y_range = y_range
    
    def generate_sgd_path(self, 
                         start_point: Tuple[float, float], 
                         lr: float = 0.1,
                         num_steps: int = 100,
                         momentum: float = 0.0) -> np.ndarray:
        """
        Generate SGD optimization path.
        
        Args:
            start_point: Starting coordinates (x, y)
            lr: Learning rate
            num_steps: Number of optimization steps
            momentum: Momentum factor (0 = no momentum)
        
        Returns:
            Array of shape (num_steps, 2) containing the path coordinates
        """
        # Initialize parameters
        x = torch.tensor([start_point[0]], requires_grad=True, dtype=torch.float32)
        y = torch.tensor([start_point[1]], requires_grad=True, dtype=torch.float32)
        
        # Track path
        path = []
        
        # Momentum buffers
        vx, vy = 0.0, 0.0
        
        for step in range(num_steps):
            # Record current position
            path.append([x.item(), y.item()])
            
            # Compute loss and gradients
            params = torch.stack([x, y])
            loss = self.objective_func(params)
            
            # Zero gradients
            if x.grad is not None:
                x.grad.zero_()
            if y.grad is not None:
                y.grad.zero_()
            
            # Backward pass
            loss.backward()
            
            # Update with momentum
            vx = momentum * vx + lr * x.grad.item()
            vy = momentum * vy + lr * y.grad.item()
            
            # Apply updates
            with torch.no_grad():
                x -= vx
                y -= vy
            
            # Clamp to bounds
            with torch.no_grad():
                x.clamp_(self.x_range[0], self.x_range[1])
                y.clamp_(self.y_range[0], self.y_range[1])
        
        return np.array(path)
    
    def generate_sgld_path(self, 
                          start_point: Tuple[float, float],
                          lr: float = 0.01,
                          temperature: float = 0.1,
                          num_steps: int = 200) -> np.ndarray:
        """
        Generate SGLD (Stochastic Gradient Langevin Dynamics) path.
        
        Args:
            start_point: Starting coordinates (x, y)
            lr: Learning rate (step size)
            temperature: Temperature parameter (controls noise level)
            num_steps: Number of sampling steps
        
        Returns:
            Array of shape (num_steps, 2) containing the path coordinates
        """
        # Initialize parameters
        x = torch.tensor([start_point[0]], requires_grad=True, dtype=torch.float32)
        y = torch.tensor([start_point[1]], requires_grad=True, dtype=torch.float32)
        
        # Track path
        path = []
        
        for step in range(num_steps):
            # Record current position
            path.append([x.item(), y.item()])
            
            # Compute loss and gradients
            params = torch.stack([x, y])
            loss = self.objective_func(params)
            
            # Zero gradients
            if x.grad is not None:
                x.grad.zero_()
            if y.grad is not None:
                y.grad.zero_()
            
            # Backward pass
            loss.backward()
            
            # SGLD update: gradient step + Gaussian noise
            with torch.no_grad():
                # Gradient step
                x -= lr * x.grad
                y -= lr * y.grad
                
                # Add Gaussian noise (Langevin dynamics)
                noise_scale = np.sqrt(2 * lr * temperature)
                x += noise_scale * torch.randn_like(x)
                y += noise_scale * torch.randn_like(y)
                
                # Clamp to bounds
                x.clamp_(self.x_range[0], self.x_range[1])
                y.clamp_(self.y_range[0], self.y_range[1])
        
        return np.array(path)


def create_objective_functions():
    """Create various objective functions for testing."""
    
    def rosenbrock_objective(params):
        """Rosenbrock function (banana function)."""
        x, y = params[0], params[1]
        a, b = 1, 100
        return (a - x)**2 + b * (y - x**2)**2
    
    def beale_objective(params):
        """Beale function."""
        x, y = params[0], params[1]
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3
    
    def gaussian_mixture_objective(params):
        """Negative log-likelihood of Gaussian mixture (for sampling)."""
        x, y = params[0], params[1]
        
        # Two Gaussian components
        # Component 1: mean=[-1, -1], cov=[[0.5, 0.2], [0.2, 0.5]]
        diff1 = torch.stack([x + 1, y + 1])
        cov1_inv = torch.tensor([[2.1739, -0.8696], [-0.8696, 2.1739]])
        log_prob1 = -0.5 * torch.dot(diff1, torch.mv(cov1_inv, diff1))
        
        # Component 2: mean=[1, 1], cov=[[0.3, -0.1], [-0.1, 0.3]]
        diff2 = torch.stack([x - 1, y - 1])
        cov2_inv = torch.tensor([[3.4483, 1.1494], [1.1494, 3.4483]])
        log_prob2 = -0.5 * torch.dot(diff2, torch.mv(cov2_inv, diff2))
        
        # Mixture (negative log-likelihood for minimization)
        mixture_prob = torch.exp(log_prob1) + 0.7 * torch.exp(log_prob2)
        return -torch.log(mixture_prob + 1e-8)
    
    def quadratic_objective(params):
        """Simple quadratic function."""
        x, y = params[0], params[1]
        return (x - 1)**2 + (y + 0.5)**2
    
    return {
        "rosenbrock": rosenbrock_objective,
        "beale": beale_objective,
        "gaussian_mixture": gaussian_mixture_objective,
        "quadratic": quadratic_objective
    }


def demo_path_comparison(objective_name: str = "gaussian_mixture"):
    """
    Demonstrate SGD vs SGLD path comparison.
    
    Args:
        objective_name: Name of objective function to use
    """
    # Get objective functions
    objectives = create_objective_functions()
    objective_func = objectives[objective_name]
    
    # Create path generator
    generator = PathGenerator(objective_func, x_range=(-3, 3), y_range=(-3, 3))
    
    # Starting point
    start_point = (-2.0, -1.5)
    
    # Generate SGD path
    sgd_path = generator.generate_sgd_path(
        start_point=start_point,
        lr=0.05,
        num_steps=100,
        momentum=0.0
    )
    
    # Generate SGLD path
    sgld_path = generator.generate_sgld_path(
        start_point=start_point,
        lr=0.02,
        temperature=0.1,
        num_steps=150
    )
    
    # Create distribution function for visualization
    distribution_funcs = {
        "rosenbrock": generate_sample_distribution("rosenbrock"),
        "beale": generate_sample_distribution("beale"),
        "gaussian_mixture": generate_sample_distribution("gaussian_mixture"),
        "quadratic": lambda X, Y: (X - 1)**2 + (Y + 0.5)**2
    }
    
    distribution_func = distribution_funcs[objective_name]
    
    # Create visualization
    scene = PathVisualization(
        distribution_func=distribution_func,
        paths=[sgd_path, sgld_path],
        path_labels=["SGD", "SGLD"],
        path_colors=["#FF6B6B", "#4ECDC4"],
        x_range=(-3, 3),
        y_range=(-3, 3),
        contour_levels=15,
        animation_time=8.0,
        show_arrows=True,
        show_dots=True
    )
    
    return scene


if __name__ == "__main__":
    # Example usage
    print("Creating SGD vs SGLD path comparison...")
    
    # Test different objective functions
    for obj_name in ["gaussian_mixture", "rosenbrock", "quadratic"]:
        print(f"\nGenerating visualization for {obj_name} function...")
        scene = demo_path_comparison(obj_name)
        
        # To render: manim path_generator.py PathVisualization -p
        # Or programmatically: scene.render()
    
    print("\nTo render animations, run:")
    print("manim path_generator.py PathVisualization -p")
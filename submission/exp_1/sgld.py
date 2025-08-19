"""
Stochastic Gradient Langevin Dynamics (SGLD) Optimizer for PyTorch

This module implements SGLD as a PyTorch optimizer subclass, following the mathematical
formulation from Welling & Teh (2011): "Bayesian Learning via Stochastic Gradient Langevin Dynamics"

SGLD Update Rule:
    Δθ_t = -ε_t ∇Ũ(θ_t) + √(2ε_t) Z_t
    
Where:
    - ε_t is the step size (learning rate) at iteration t  
    - ∇Ũ(θ_t) is the noisy gradient estimate from mini-batch
    - Z_t ~ N(0, I) is Gaussian noise
    - The noise scale √(2ε_t) ensures proper diffusion scaling
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, Optional


class SGLD(Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) optimizer.
    
    This optimizer implements the SGLD algorithm which combines stochastic gradient descent
    with Langevin dynamics to enable Bayesian posterior sampling while optimizing.
    
    The algorithm transitions from optimization (when step size is large) to sampling 
    (when step size becomes small), providing both point estimates and uncertainty quantification.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate / step size (default: 1e-2)
        temperature (float, optional): temperature parameter controlling noise level (default: 1.0)
        momentum (float, optional): momentum factor (0 = no momentum) (default: 0.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        noise_decay (float, optional): decay factor for noise over time (default: 1.0)
        min_lr (float, optional): minimum learning rate for polynomial decay (default: 1e-6)
        lr_decay (float, optional): learning rate decay exponent (default: 0.55)
        
    Note:
        - For convergence to the true posterior, the step size should satisfy Robbins-Monro conditions:
          Σ ε_t = ∞ and Σ ε_t² < ∞
        - A common choice is polynomial decay: ε_t = a(b + t)^(-γ) with γ ∈ (0.5, 1]
        - Set temperature=0 to recover standard SGD behavior
    """
    
    def __init__(self, 
                 params, 
                 lr: float = 1e-2,
                 temperature: float = 1.0,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 noise_decay: float = 1.0,
                 min_lr: float = 1e-6,
                 lr_decay: float = 0.55):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= noise_decay <= 1.0:
            raise ValueError(f"Invalid noise_decay value: {noise_decay}")
        if not 0.0 < lr_decay <= 1.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= min_lr <= lr:
            raise ValueError(f"Invalid min_lr value: {min_lr}")

        defaults = dict(
            lr=lr, 
            temperature=temperature,
            momentum=momentum,
            weight_decay=weight_decay,
            noise_decay=noise_decay,
            min_lr=min_lr,
            lr_decay=lr_decay
        )
        super(SGLD, self).__init__(params, defaults)
        
        # Initialize global step counter
        self.state['global_step'] = 0

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Increment global step counter
        self.state['global_step'] += 1
        global_step = self.state['global_step']

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            temperature = group['temperature']
            noise_decay = group['noise_decay']
            base_lr = group['lr']
            min_lr = group['min_lr']
            lr_decay = group['lr_decay']

            # Apply polynomial decay schedule: lr_t = max(min_lr, base_lr * (1 + t)^(-lr_decay))
            current_lr = max(min_lr, base_lr * (1 + global_step) ** (-lr_decay))
            
            # Apply noise decay over time
            current_temperature = temperature * (noise_decay ** global_step)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p)

                buf = param_state['momentum_buffer']

                # Apply momentum (if specified)
                if momentum != 0:
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Generate Langevin noise: √(2 * lr * temperature) * N(0, I)
                if current_temperature > 0:
                    noise_scale = math.sqrt(2 * current_lr * current_temperature)
                    noise = torch.randn_like(p) * noise_scale
                else:
                    noise = 0

                # SGLD update: θ_{t+1} = θ_t - lr * ∇L + noise
                p.add_(grad, alpha=-current_lr).add_(noise)

        return loss

    def get_current_lr(self) -> float:
        """Get the current learning rate after decay."""
        global_step = self.state['global_step']
        group = self.param_groups[0]  # Assume all groups have same lr schedule
        base_lr = group['lr']
        min_lr = group['min_lr']
        lr_decay = group['lr_decay']
        
        return max(min_lr, base_lr * (1 + global_step) ** (-lr_decay))

    def get_current_temperature(self) -> float:
        """Get the current temperature after decay."""
        global_step = self.state['global_step']
        group = self.param_groups[0]  # Assume all groups have same temperature schedule
        temperature = group['temperature']
        noise_decay = group['noise_decay']
        
        return temperature * (noise_decay ** global_step)

    def set_temperature(self, temperature: float):
        """Set temperature for all parameter groups."""
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        
        for group in self.param_groups:
            group['temperature'] = temperature

    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        for group in self.param_groups:
            group['lr'] = lr

    def get_info(self) -> Dict[str, Any]:
        """Get current optimizer state information."""
        return {
            'global_step': self.state['global_step'],
            'current_lr': self.get_current_lr(),
            'current_temperature': self.get_current_temperature(),
            'param_groups': len(self.param_groups),
            'total_params': sum(len(group['params']) for group in self.param_groups)
        }


# Alternative step size schedules
class SGLDWithCustomSchedule(SGLD):
    """
    SGLD with custom step size and temperature schedules.
    
    This variant allows for more flexible scheduling beyond polynomial decay.
    """
    
    def __init__(self, params, lr=1e-2, temperature=1.0, momentum=0.0, weight_decay=0.0,
                 lr_schedule=None, temperature_schedule=None):
        """
        Arguments:
            lr_schedule (callable, optional): Function that takes step number and returns learning rate
            temperature_schedule (callable, optional): Function that takes step number and returns temperature
        """
        super().__init__(params, lr, temperature, momentum, weight_decay)
        self.lr_schedule = lr_schedule
        self.temperature_schedule = temperature_schedule

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with custom schedules."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['global_step'] += 1
        global_step = self.state['global_step']

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            
            # Use custom schedules if provided
            if self.lr_schedule is not None:
                current_lr = self.lr_schedule(global_step)
            else:
                current_lr = group['lr']
                
            if self.temperature_schedule is not None:
                current_temperature = self.temperature_schedule(global_step)
            else:
                current_temperature = group['temperature']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p)

                buf = param_state['momentum_buffer']

                if momentum != 0:
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Generate Langevin noise
                if current_temperature > 0:
                    noise_scale = math.sqrt(2 * current_lr * current_temperature)
                    noise = torch.randn_like(p) * noise_scale
                else:
                    noise = 0

                # SGLD update
                p.add_(grad, alpha=-current_lr).add_(noise)

        return loss


# Convenience function for common schedules
def polynomial_schedule(base_lr: float, decay_rate: float = 0.55, min_lr: float = 1e-6):
    """Create a polynomial decay schedule function."""
    def schedule(step):
        return max(min_lr, base_lr * (1 + step) ** (-decay_rate))
    return schedule


def cosine_annealing_schedule(base_lr: float, min_lr: float = 1e-6, period: int = 1000):
    """Create a cosine annealing schedule function."""
    def schedule(step):
        cycle_position = (step % period) / period
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * cycle_position))
    return schedule


def exponential_schedule(base_lr: float, decay_rate: float = 0.99, min_lr: float = 1e-6):
    """Create an exponential decay schedule function."""
    def schedule(step):
        return max(min_lr, base_lr * (decay_rate ** step))
    return schedule

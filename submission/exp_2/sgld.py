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
        temperature_decay (float, optional): decay factor for temperature over time (default: 1.0)
        
    Note:
        - This implementation uses a constant learning rate and a decaying temperature
          (annealing), which transitions the optimizer from an exploration phase to a
          posterior sampling phase.
        - Set temperature=0 to recover standard SGD behavior
    """
    
    def __init__(self, 
                 params, 
                 lr: float = 1e-2,
                 temperature: float = 1.0,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 temperature_decay: float = 1.0):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= temperature_decay <= 1.0:
            raise ValueError(f"Invalid temperature_decay value: {temperature_decay}")

        defaults = dict(
            lr=lr, 
            temperature=temperature,
            momentum=momentum,
            weight_decay=weight_decay,
            temperature_decay=temperature_decay
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
            temperature_decay = group['temperature_decay']
            current_lr = group['lr']  # Use constant learning rate
            
            # Apply temperature decay over time
            current_temperature = temperature * (temperature_decay ** global_step)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Apply weight decay (modifies gradient in-place)
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

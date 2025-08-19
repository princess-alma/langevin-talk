"""
Simple MLP for moons classification experiment.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP for 2D moons classification - challenging single hidden layer."""
    
    def __init__(self, input_size=2, hidden_size=5, num_classes=2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def get_moons_model():
    """Get the model for moons classification - small and challenging."""
    return SimpleMLP(input_size=2, hidden_size=1, num_classes=2)

"""
Fast ConvNet for CIFAR-10 SGLD Experiment

Simple, efficient ConvNet architecture optimized for CIFAR-10.
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Separable convolution for efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FastCIFAR10Net(nn.Module):
    """Simple ConvNet for CIFAR-10."""
    
    def __init__(self, base_width=16):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, base_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            SeparableConv2d(base_width, base_width*2, 3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            SeparableConv2d(base_width*2, base_width*4, 3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(base_width*4, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_cifar10_model():
    """Get the model for CIFAR-10 experiment."""
    return FastCIFAR10Net(base_width=16)

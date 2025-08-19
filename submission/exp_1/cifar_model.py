"""
CIFAR-10 ConvNet Model Definition

ConvNet architecture optimized for CIFAR-10 classification (32x32 RGB images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10ConvNet(nn.Module):
    """ConvNet optimized for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CIFAR10ConvNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # After conv layers: 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class SimpleCIFAR10Net(nn.Module):
    """Simpler ConvNet for CIFAR-10 (based on your original style)."""
    
    def __init__(self, num_classes=10):
        super(SimpleCIFAR10Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling operations: 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv + Pool layers
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Example usage and testing
if __name__ == "__main__":
    print("Testing CIFAR-10 models...")
    
    # Test the advanced model
    model1 = CIFAR10ConvNet()
    print(f"\nCIFAR10ConvNet:")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Test with sample input
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    output1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output1.shape}")
    
    # Test the simple model
    model2 = SimpleCIFAR10Net()
    print(f"\nSimpleCIFAR10Net:")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    output2 = model2(x)
    print(f"Output shape: {output2.shape}")
    
    print(f"\nBoth models work correctly!")

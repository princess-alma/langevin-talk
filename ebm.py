import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

class EnergyBasedModel(nn.Module):
    """Energy-Based Model with CNN encoder outputting a single scalar energy value"""
    
    def __init__(self, input_channels=1, hidden_dim=128):
        super(EnergyBasedModel, self).__init__()
        
        # CNN Encoder layers
        self.conv1 = spectral_norm(nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))

        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate the size after convolutions for MNIST (28x28)
        # After conv2: 14x14, after conv3: 7x7, after conv4: 4x4
        self.feature_size = 256

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        
        # Fully connected layers to output scalar energy
        self.fc1 = nn.Linear(self.feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)  # Output single scalar energy
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through the energy model
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            energy: Scalar energy values of shape (batch_size, 1)
        """
        # CNN encoder
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        
        # Flatten for fully connected layers
        x = self.global_pool(x)#x.view(x.size(0), -1)
        #print(f"Shape after global pool: {x.shape}")
        #x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)  # [batch, 256]
        # Fully connected layers with dropout
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))
        x = self.dropout(x)
        
        # Output energy (no activation - can be positive or negative)
        energy = self.fc3(x)
        
        return energy
    
    def energy(self, x):
        """Alias for forward pass - more semantically clear for EBM"""
        return self.forward(x)


class SimpleEBM(nn.Module):
    """Simpler Energy-Based Model for faster training"""
    
    def __init__(self, input_channels=1):
        super(SimpleEBM, self).__init__()
        
        # Simpler CNN encoder
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)  # 28->14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14->7
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 7->4 (with padding)
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output layer
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Forward pass through the simple energy model
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            energy: Scalar energy values of shape (batch_size, 1)
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Output energy
        energy = self.fc(x)
        
        return energy
    
    def energy(self, x):
        """Alias for forward pass"""
        return self.forward(x)


def test_ebm():
    """Test function to verify the EBM models work correctly"""
    # Create test input (batch_size=4, channels=1, height=28, width=28)
    test_input = torch.randn(4, 1, 28, 28)
    
    # Test full EBM
    print("Testing EnergyBasedModel:")
    ebm = EnergyBasedModel(input_channels=1, hidden_dim=128)
    energy_output = ebm(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Energy output shape: {energy_output.shape}")
    print(f"Energy values: {energy_output.squeeze().detach().numpy()}")
    print(f"Model parameters: {sum(p.numel() for p in ebm.parameters()):,}")
    
    print("\nTesting SimpleEBM:")
    simple_ebm = SimpleEBM(input_channels=1)
    simple_energy = simple_ebm(test_input)
    print(f"Simple energy output shape: {simple_energy.shape}")
    print(f"Simple energy values: {simple_energy.squeeze().detach().numpy()}")
    print(f"Simple model parameters: {sum(p.numel() for p in simple_ebm.parameters()):,}")


def load_trained_ebm(model_path="trained_ebm.pth", device='cpu'):
    """
    Load a trained EBM model from file
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
    Returns:
        Loaded model and training info
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    model = EnergyBasedModel(
        input_channels=model_config['input_channels'],
        hidden_dim=model_config['hidden_dim']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    if 'training_params' in checkpoint:
        print(f"Training params: {checkpoint['training_params']}")
    
    return model, checkpoint.get('training_params', {})


if __name__ == "__main__":
    test_ebm()
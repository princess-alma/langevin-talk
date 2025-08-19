"""
Fast ConvNet Model Definition

Efficient ConvNet architecture using separable convolutions for fast training.
Works with both MNIST (1 channel) and CIFAR-10 (3 channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Separable convolution: depthwise + pointwise convolution for efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FastConvNet(nn.Module):
    """Fast ConvNet using separable convolutions for efficient training."""
    
    def __init__(self, input_channels=3, num_classes=10, dropout_rate=0.3, base_width=32):
        super(FastConvNet, self).__init__()
        
        # Initial regular convolution
        self.conv1 = nn.Conv2d(input_channels, base_width, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        
        # Separable convolution blocks for efficiency
        self.sep_conv1 = SeparableConv2d(base_width, base_width * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_width * 2)
        
        self.sep_conv2 = SeparableConv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_width * 4)
        
        self.sep_conv3 = SeparableConv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_width * 8)
        
        # Global average pooling instead of large FC layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Small classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_width * 8, num_classes)
        
        # Initialize weights for faster convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for faster convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Block 1: Regular conv + BN + ReLU + Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16 (or 28x28 -> 14x14 for MNIST)
        
        # Block 2: Separable conv + BN + ReLU + Pool  
        x = F.relu(self.bn2(self.sep_conv1(x)))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8 (or 14x14 -> 7x7)
        
        # Block 3: Separable conv + BN + ReLU + Pool
        x = F.relu(self.bn3(self.sep_conv2(x)))
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4 (or 7x7 -> 3x3)
        
        # Block 4: Separable conv + BN + ReLU
        x = F.relu(self.bn4(self.sep_conv3(x)))
        
        # Global average pooling to reduce parameters
        x = self.global_pool(x)  # -> Nx(base_width*8)x1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# Convenience functions for different input types
def get_mnist_model(num_classes=10, dropout_rate=0.3):
    """Get model configured for MNIST (1 channel input)."""
    return FastConvNet(input_channels=1, num_classes=num_classes, 
                      dropout_rate=dropout_rate, base_width=32)


def get_cifar10_model(num_classes=10, dropout_rate=0.3):
    """Get model configured for CIFAR-10 (3 channel input)."""
    return FastConvNet(input_channels=3, num_classes=num_classes, 
                      dropout_rate=dropout_rate, base_width=32)


# Backward compatibility alias
MNISTConvNet = get_mnist_model


class EnsembleModel:
    """Ensemble of multiple models for prediction."""
    
    def __init__(self, model_class, model_kwargs, device):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.device = device
        self.models = []
    
    def add_model(self, state_dict):
        """Add a model to the ensemble."""
        model = self.model_class(**self.model_kwargs)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.models.append(model)
    
    def predict(self, data_loader):
        """Make ensemble predictions."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                # Get predictions from all models
                batch_predictions = []
                for model in self.models:
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs)
                
                # Average predictions
                ensemble_probs = torch.stack(batch_predictions).mean(dim=0)
                all_predictions.append(ensemble_probs)
                all_targets.append(target)
        
        return torch.cat(all_predictions), torch.cat(all_targets)
    
    def evaluate(self, data_loader):
        """Evaluate ensemble accuracy."""
        predictions, targets = self.predict(data_loader)
        _, predicted_classes = torch.max(predictions, 1)
        
        accuracy = 100.0 * (predicted_classes.cpu() == targets).float().mean().item()
        
        # Calculate ensemble uncertainty (entropy)
        entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        avg_uncertainty = entropies.mean().item()
        
        return accuracy, avg_uncertainty


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size):
    """Print a summary of the model architecture."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test with dummy input to get output size
    if len(input_size) == 3:  # CHW format
        dummy_input = torch.randn(1, *input_size)
    else:  # Assume batch size included
        dummy_input = torch.randn(input_size)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input size: {list(dummy_input.shape)}")
    print(f"Output size: {list(output.shape)}")
    print(f"Output classes: {output.shape[-1]}")


if __name__ == "__main__":
    # Test the models
    print("Testing Fast ConvNet models...")
    
    # Test MNIST model
    print("\n=== MNIST Model ===")
    mnist_model = get_mnist_model()
    model_summary(mnist_model, (1, 28, 28))
    
    # Test CIFAR-10 model  
    print("\n=== CIFAR-10 Model ===")
    cifar_model = get_cifar10_model()
    model_summary(cifar_model, (3, 32, 32))
    
    # Compare parameter counts
    print(f"\nParameter comparison:")
    print(f"MNIST model: {count_parameters(mnist_model):,} parameters")
    print(f"CIFAR-10 model: {count_parameters(cifar_model):,} parameters")
    
    print("\nModels tested successfully!")

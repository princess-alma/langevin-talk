# SGLD Experiment: SGD vs SGLD Comparison

This experiment compares SGD and SGLD optimizers on MNIST and CIFAR-10 classification using a fast ConvNet with separable convolutions.

## Files

- `model.py`: Contains the FastConvNet architecture with separable convolutions and EnsembleModel class
- `sgld.py`: SGLD optimizer implementation with Langevin dynamics
- `experiment.py`: Main experiment script with training functions and evaluation
- `cifar10_loader.py`: CIFAR-10 data loading utilities

## Structure

- `train_sgd()`: Training function using standard SGD optimizer
- `train_sgld()`: Training function using SGLD optimizer  
- `run_experiment()`: Main function that runs both trainings and compares results

## Usage

```bash
cd exp_1

# Run with MNIST (default)
python experiment.py
python experiment.py mnist

# Run with CIFAR-10
python experiment.py cifar10
```

## Key Features

- **Fast Architecture**: Uses separable convolutions for efficient training
- **Multi-Dataset**: Supports both MNIST and CIFAR-10
- **Fair comparison**: Both optimizers save top N models based on validation accuracy
- **Ensemble evaluation**: Creates ensembles from best models of each optimizer
- **Uncertainty quantification**: Compares uncertainty estimates between methods
- **Training visualization**: Plots training curves for both optimizers

## Model Architecture

- Separable convolutions (depthwise + pointwise) for efficiency
- Global average pooling to reduce parameters
- Batch normalization for stable training
- Optimized initialization for faster convergence

## Expected Insights

- SGLD should provide better uncertainty quantification due to stochastic exploration
- SGLD ensembles may show better diversity and calibration
- Both methods should achieve comparable accuracy
- Separable convolutions enable much faster training

# Moons Dataset: SGD vs SGLD Comparison

This experiment demonstrates the differences between Stochastic Gradient Descent (SGD) and Stochastic Gradient Langevin Dynamics (SGLD) on the challenging two moons classification problem.

## Experiment Overview

**Problem**: Binary classification on two interleaving crescent-shaped clusters  
**Model**: Simple MLP with 1 hidden layer (2→5→2, 37 parameters)  
**Key Feature**: Both optimizers start with identical weights for fair comparison

## Files

- `experiment.py` - Main experiment script
- `model.py` - Neural network architecture  
- `moons_loader.py` - Dataset loading and preprocessing
- `moons_comparison_final.png` - Results visualization

## Key Settings

```python
# Optimal hyperparameters discovered through experimentation
NUM_EPOCHS = 2000
LEARNING_RATE = 0.01
SGLD_TEMPERATURE = 0.5
SGLD_NOISE_DECAY = 0.9996  # Very slow decay for extended exploration
BATCH_SIZE = 32
```

## Expected Results

- **SGD**: Stable training, may get stuck in local minima
- **SGLD**: Initial exploration phase, then convergence to competitive performance
- **Key Insight**: SGLD's exploration helps with non-linear decision boundaries

## Running the Experiment

```bash
python experiment.py
```

Results are saved as `moons_comparison_final.png` showing training/validation loss and accuracy curves for both optimizers.

## Architecture Details

**SimpleMLP**: 
- Input: 2D coordinates (x1, x2)
- Hidden: 5 neurons with ReLU activation  
- Output: 2 classes (binary classification)
- Total parameters: 37 (2×5 + 5 + 5×2 + 2)

**Why this is challenging**: With only 5 hidden neurons, the model must learn an efficient representation of the curved decision boundary between the two moons.

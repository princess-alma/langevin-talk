"""
CIFAR-10 ConvNet Experiment: SGD vs SGLD Fair Ensemble Comparison

This experiment provides a fair comparison between SGD and SGLD by:
1. Training identical ConvNets with both optimizers on CIFAR-10
2. Saving the top N epochs from each optimizer based on validation accuracy
3. Creating ensembles from the best N models for both SGD and SGLD
4. Benchmarking all approaches: single best SGD, single best SGLD, SGD ensemble, SGLD ensemble
"""

import sys
import os
sys.path.append('../..')  # Add parent directory to path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
import time

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    print("tqdm not available - using simple progress indication")
    HAVE_TQDM = False
    def tqdm(iterable, desc="", leave=True):
        return iterable

# Import our components
from model import FastConvNet, EnsembleModel, get_cifar10_model
from sgld import SGLD
from cifar10_loader import load_cifar10_data


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy and loss on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(data_loader)
    
    return accuracy, avg_loss


def train_sgd(model, train_loader, val_loader, num_epochs, device, save_top_n=5):
    """
    Train a model using SGD optimizer and save the top N epochs based on validation accuracy.
    
    Returns:
        best_models: List of (epoch, model_state_dict, val_acc) for top N epochs
        training_history: Dictionary with loss and accuracy history
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize SGD optimizer with higher lr for CIFAR-10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Track best models
    best_models = []  # List of (epoch, state_dict, val_acc)
    
    print(f"\nTraining with SGD...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        if HAVE_TQDM:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        else:
            pbar = train_loader
            print(f'Epoch {epoch+1}/{num_epochs}: ', end='', flush=True)
            
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar or print progress
            if HAVE_TQDM and batch_idx % 50 == 0:
                current_acc = 100.0 * train_correct / train_total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            elif not HAVE_TQDM and batch_idx % 50 == 0:
                print('.', end='', flush=True)
        
        if not HAVE_TQDM:
            print()  # New line after progress dots
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        train_acc = 100.0 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation
        val_acc, val_loss_avg = evaluate_model(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        # Check if this model should be saved (top N validation accuracy)
        model_info = (epoch, copy.deepcopy(model.state_dict()), val_acc)
        best_models.append(model_info)
        
        # Keep only top N models
        best_models.sort(key=lambda x: x[2], reverse=True)  # Sort by val_acc
        if len(best_models) > save_top_n:
            best_models = best_models[:save_top_n]
        
        # Print epoch results
        print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:6.2f}%, "
              f"Val Acc: {val_acc:6.2f}%, Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}")
    
    print(f"\nTop {len(best_models)} epochs for SGD:")
    for i, (epoch, _, val_acc) in enumerate(best_models):
        print(f"  {i+1}. Epoch {epoch+1}: {val_acc:.2f}%")
    
    return best_models, history


def train_sgld(model, train_loader, val_loader, num_epochs, device, save_top_n=5):
    """
    Train a model using SGLD optimizer and save the top N epochs based on validation accuracy.
    
    Returns:
        best_models: List of (epoch, model_state_dict, val_acc) for top N epochs
        training_history: Dictionary with loss and accuracy history
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize SGLD optimizer
    optimizer = SGLD(
        model.parameters(),
        lr=0.1,                 # Higher LR for CIFAR-10
        temperature=0.01,       # Small temperature for stability
        momentum=0.9,
        weight_decay=1e-4,
        lr_decay=0.5,          # Moderate decay
        min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Track best models
    best_models = []  # List of (epoch, state_dict, val_acc)
    
    print(f"\nTraining with SGLD...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        if HAVE_TQDM:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        else:
            pbar = train_loader
            print(f'Epoch {epoch+1}/{num_epochs}: ', end='', flush=True)
            
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar or print progress
            if HAVE_TQDM and batch_idx % 50 == 0:
                current_acc = 100.0 * train_correct / train_total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            elif not HAVE_TQDM and batch_idx % 50 == 0:
                print('.', end='', flush=True)
        
        if not HAVE_TQDM:
            print()  # New line after progress dots
        
        # Calculate epoch metrics
        train_acc = 100.0 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation
        val_acc, val_loss_avg = evaluate_model(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        # Check if this model should be saved (top N validation accuracy)
        model_info = (epoch, copy.deepcopy(model.state_dict()), val_acc)
        best_models.append(model_info)
        
        # Keep only top N models
        best_models.sort(key=lambda x: x[2], reverse=True)  # Sort by val_acc
        if len(best_models) > save_top_n:
            best_models = best_models[:save_top_n]
        
        # Print epoch results
        sgld_info = optimizer.get_info()
        optimizer_info = f" (LR: {sgld_info['current_lr']:.6f}, Temp: {sgld_info['current_temperature']:.6f})"
        
        print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:6.2f}%, "
              f"Val Acc: {val_acc:6.2f}%, Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}{optimizer_info}")
    
    print(f"\nTop {len(best_models)} epochs for SGLD:")
    for i, (epoch, _, val_acc) in enumerate(best_models):
        print(f"  {i+1}. Epoch {epoch+1}: {val_acc:.2f}%")
    
    return best_models, history


def plot_training_curves(sgd_history, sgld_history, save_path='training_curves.png'):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs_sgd = range(1, len(sgd_history['train_loss']) + 1)
    epochs_sgld = range(1, len(sgld_history['train_loss']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs_sgd, sgd_history['train_loss'], 'b-', label='SGD')
    axes[0, 0].plot(epochs_sgld, sgld_history['train_loss'], 'r-', label='SGLD')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation Loss
    axes[0, 1].plot(epochs_sgd, sgd_history['val_loss'], 'b-', label='SGD')
    axes[0, 1].plot(epochs_sgld, sgld_history['val_loss'], 'r-', label='SGLD')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training Accuracy
    axes[1, 0].plot(epochs_sgd, sgd_history['train_acc'], 'b-', label='SGD')
    axes[1, 0].plot(epochs_sgd, sgld_history['train_acc'], 'r-', label='SGLD')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Accuracy
    axes[1, 1].plot(epochs_sgd, sgd_history['val_acc'], 'b-', label='SGD')
    axes[1, 1].plot(epochs_sgld, sgld_history['val_acc'], 'r-', label='SGLD')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")


def run_experiment():
    """Main experiment function for CIFAR-10."""
    print("Fast ConvNet Demo: SGD vs SGLD Fair Ensemble Comparison on CIFAR-10")
    print("=" * 70)
    
    # Configuration
    BATCH_SIZE = 256  # Larger batch for CIFAR-10
    NUM_EPOCHS = 20   # Fewer epochs for CIFAR-10
    TOP_N_MODELS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Dataset: CIFAR-10")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Ensemble size: {TOP_N_MODELS}")
    print()
    
    # Load CIFAR-10 data
    print("Loading CIFAR-10 data...")
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Train SGLD model
    print("\n" + "="*60)
    sgld_model = get_cifar10_model()
    sgld_best_models, sgld_history = train_sgld(
        sgld_model, train_loader, val_loader, NUM_EPOCHS, DEVICE, TOP_N_MODELS
    )
    
    # Train SGD model
    print("\n" + "="*60)
    sgd_model = get_cifar10_model()
    sgd_best_models, sgd_history = train_sgd(
        sgd_model, train_loader, val_loader, NUM_EPOCHS, DEVICE, TOP_N_MODELS
    )
    
    # Create ensembles
    print("\n" + "="*60)
    print("Creating Ensembles...")
    
    # SGD ensemble
    sgd_ensemble = EnsembleModel(get_cifar10_model, {}, DEVICE)
    for epoch, state_dict, val_acc in sgd_best_models:
        sgd_ensemble.add_model(state_dict)
    
    # SGLD ensemble
    sgld_ensemble = EnsembleModel(get_cifar10_model, {}, DEVICE)
    for epoch, state_dict, val_acc in sgld_best_models:
        sgld_ensemble.add_model(state_dict)
    
    # Evaluate single best models
    print("\nEvaluating Single Best Models...")
    
    # Best SGD model
    best_sgd_model = get_cifar10_model()
    best_sgd_model.load_state_dict(sgd_best_models[0][1])  # Best is first after sorting
    sgd_single_acc, _ = evaluate_model(best_sgd_model, test_loader, DEVICE)
    
    # Best SGLD model  
    best_sgld_model = get_cifar10_model()
    best_sgld_model.load_state_dict(sgld_best_models[0][1])
    sgld_single_acc, _ = evaluate_model(best_sgld_model, test_loader, DEVICE)
    
    # Evaluate ensembles
    print("Evaluating Ensembles...")
    sgd_ensemble_acc, sgd_uncertainty = sgd_ensemble.evaluate(test_loader)
    sgld_ensemble_acc, sgld_uncertainty = sgld_ensemble.evaluate(test_loader)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nSingle Model Performance:")
    print(f"  Best SGD Model:     {sgd_single_acc:.2f}%")
    print(f"  Best SGLD Model:    {sgld_single_acc:.2f}%")
    
    print(f"\nEnsemble Performance ({TOP_N_MODELS} models):")
    print(f"  SGD Ensemble:       {sgd_ensemble_acc:.2f}%")
    print(f"  SGLD Ensemble:      {sgld_ensemble_acc:.2f}%")
    
    print(f"\nUncertainty Analysis (Average Entropy):")
    print(f"  SGD Ensemble:       {sgd_uncertainty:.4f}")
    print(f"  SGLD Ensemble:      {sgld_uncertainty:.4f}")
    
    print(f"\nImprovement Analysis:")
    sgd_improvement = sgd_ensemble_acc - sgd_single_acc
    sgld_improvement = sgld_ensemble_acc - sgld_single_acc
    print(f"  SGD:  Ensemble vs Single: {sgd_improvement:+.2f}%")
    print(f"  SGLD: Ensemble vs Single: {sgld_improvement:+.2f}%")
    
    ensemble_advantage = sgld_ensemble_acc - sgd_ensemble_acc
    print(f"  SGLD vs SGD Ensemble:     {ensemble_advantage:+.2f}%")
    
    # Plot training curves
    plot_training_curves(sgd_history, sgld_history, 'cifar10_training_curves.png')
    
    # Summary insights
    print(f"\n" + "="*60)
    print("INSIGHTS")
    print("="*60)
    
    if sgld_ensemble_acc > sgd_ensemble_acc:
        print("✓ SGLD ensemble outperforms SGD ensemble")
    else:
        print("✗ SGD ensemble outperforms SGLD ensemble")
    
    if sgld_uncertainty > sgd_uncertainty:
        print("✓ SGLD provides higher uncertainty estimates (better calibration)")
    else:
        print("✗ SGD provides higher uncertainty estimates")
        
    if sgld_improvement > sgd_improvement:
        print("✓ SGLD benefits more from ensembling")
    else:
        print("✗ SGD benefits more from ensembling")
    
    print(f"\nKey Takeaways:")
    print(f"• Fair comparison: Both optimizers use top {TOP_N_MODELS} epochs")
    print(f"• SGLD exploration helps with ensemble diversity")
    print(f"• Uncertainty quantification differs between methods")
    print(f"• Fast separable convolutions for efficient training")


if __name__ == "__main__":
    print("Running CIFAR-10 experiment: SGD vs SGLD")
    print("Using fast separable convolutions for efficient training")
    print()
    
    run_experiment()

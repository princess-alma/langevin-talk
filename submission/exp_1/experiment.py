"""
Simple CIFAR-10 Experiment: SGD vs SGLD

Clean comparison of SGD and SGLD optimizers on CIFAR-10.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from model import get_cifar10_model
from sgld import SGLD
from cifar10_loader import load_cifar10_data

# Check for tqdm
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    """Train a model and return training history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        if HAVE_TQDM:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        else:
            pbar = train_loader
            print(f'Epoch {epoch+1}/{num_epochs}: ', end='', flush=True)
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if HAVE_TQDM:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Record history
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def plot_training_curves(sgd_history, sgld_history):
    """Plot training curves for comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(sgd_history['train_loss']) + 1)
    
    # Training loss
    ax1.plot(epochs, sgd_history['train_loss'], 'b-', label='SGD')
    ax1.plot(epochs, sgld_history['train_loss'], 'r-', label='SGLD')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation loss
    ax2.plot(epochs, sgd_history['val_loss'], 'b-', label='SGD')
    ax2.plot(epochs, sgld_history['val_loss'], 'r-', label='SGLD')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Training accuracy
    ax3.plot(epochs, sgd_history['train_acc'], 'b-', label='SGD')
    ax3.plot(epochs, sgld_history['train_acc'], 'r-', label='SGLD')
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Validation accuracy
    ax4.plot(epochs, sgd_history['val_acc'], 'b-', label='SGD')
    ax4.plot(epochs, sgld_history['val_acc'], 'r-', label='SGLD')
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('sgd_vs_sgld_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_experiment():
    """Run the complete CIFAR-10 experiment."""
    print("CIFAR-10 Experiment: SGD vs SGLD")
    print("=" * 40)
    
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()
    
    # Load data
    print("Loading CIFAR-10 data...")
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()

    print("\nTraining with SGLD...")
    sgld_model = get_cifar10_model().to(DEVICE)
    sgld_optimizer = SGLD(sgld_model.parameters(), lr=0.01, temperature=0.01, momentum=0.9, weight_decay=1e-4)
    sgld_history = train_model(sgld_model, train_loader, val_loader, sgld_optimizer, NUM_EPOCHS, DEVICE)
   
    
    # Train with SGD
    print("Training with SGD...")
    sgd_model = get_cifar10_model().to(DEVICE)
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sgd_history = train_model(sgd_model, train_loader, val_loader, sgd_optimizer, NUM_EPOCHS, DEVICE)
    
 
    # Plot results
    print("\nPlotting training curves...")
    plot_training_curves(sgd_history, sgld_history)
    
    # Final evaluation
    print("\nFinal Test Set Evaluation:")
    sgd_test_acc = evaluate_model(sgd_model, test_loader, DEVICE)
    sgld_test_acc = evaluate_model(sgld_model, test_loader, DEVICE)
    
    print(f"SGD Test Accuracy: {sgd_test_acc:.2f}%")
    print(f"SGLD Test Accuracy: {sgld_test_acc:.2f}%")
    
    return {
        'sgd_history': sgd_history,
        'sgld_history': sgld_history,
        'sgd_test_acc': sgd_test_acc,
        'sgld_test_acc': sgld_test_acc
    }


if __name__ == "__main__":
    results = run_experiment()

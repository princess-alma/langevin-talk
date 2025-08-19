"""
Moons Experiment: SGD vs SGLD

Clean experiment comparing SGD vs SGLD on the two moons dataset.
This demonstrates SGLD's exploration advantages on non-linear decision boundaries.
"""

import sys
sys.path.append('..')  # To access shared sgld.py

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from model import get_moons_model  # Model architecture
from sgld import SGLD
from moons_loader import load_moons_data
import random


def init_weights(model, seed):
    """Initialize model weights with a given seed."""
    torch.manual_seed(seed)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, optimizer_name):
    """Train a model and return training history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining with {optimizer_name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
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
        
        if (epoch + 1) % 100 == 0:
            # Show current SGLD parameters if it's SGLD
            if hasattr(optimizer, 'get_current_temperature'):
                current_temp = optimizer.get_current_temperature()
                current_lr = optimizer.get_current_lr()
                print(f'{optimizer_name} Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}% (LR: {current_lr:.4f}, Temp: {current_temp:.4f})')
            else:
                print(f'{optimizer_name} Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')
    
    training_time = time.time() - start_time
    print(f'{optimizer_name} Training Time: {training_time:.1f}s')
    
    return history, training_time


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


def plot_comparison(sgd_history, sgld_history):
    """Plot training comparison with all metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(sgd_history['train_loss']) + 1)
    
    # Training Loss
    ax1.plot(epochs, sgd_history['train_loss'], 'b-', label='SGD', linewidth=2)
    ax1.plot(epochs, sgld_history['train_loss'], 'r-', label='SGLD', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2.plot(epochs, sgd_history['val_loss'], 'b-', label='SGD', linewidth=2)
    ax2.plot(epochs, sgld_history['val_loss'], 'r-', label='SGLD', linewidth=2)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3.plot(epochs, sgd_history['train_acc'], 'b-', label='SGD', linewidth=2)
    ax3.plot(epochs, sgld_history['train_acc'], 'r-', label='SGLD', linewidth=2)
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax4.plot(epochs, sgd_history['val_acc'], 'b-', label='SGD', linewidth=2)
    ax4.plot(epochs, sgld_history['val_acc'], 'r-', label='SGLD', linewidth=2)
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moons_comparison_final.png', dpi=150, bbox_inches='tight')
    print("📊 Plot saved as 'moons_comparison_final.png'")
    plt.show()


def run_experiment():
    """Run the complete moons experiment with optimal settings."""
    print("Moons Dataset Experiment: SGD vs SGLD")
    print("=" * 42)
    
    # Optimal configuration based on our experiments
    BATCH_SIZE = 32
    NUM_EPOCHS = 3000  # Extended training for clear differences
    LR = 0.001  # Higher LR works well for small networks
    TEMPERATURE = 0.6  # Moderate initial temperature
    NOISE_DECAY = 0.9995  # Much slower decay to maintain exploration through all 2000 epochs
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"SGLD Temperature: {TEMPERATURE}")
    print(f"SGLD Noise Decay: {NOISE_DECAY}")
    print()
    
    # Load data
    print("Loading moons data...")
    train_loader, val_loader, test_loader = load_moons_data(
        n_samples=1000, noise=0.1, batch_size=BATCH_SIZE
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Generate random seed for this run (same for both models, different across runs)
    run_seed = random.randint(1, 10000)
    print(f"\nRun seed: {run_seed}")
    print("Initializing models with identical weights for this run...")
    
    # Train with SGD
    sgd_model = get_moons_model().to(DEVICE)
    init_weights(sgd_model, seed=run_seed)  # Random seed for this run
    
    # Train with SGLD (same initialization for fair comparison!)
    sgld_model = get_moons_model().to(DEVICE)
    init_weights(sgld_model, seed=run_seed)  # Same seed as SGD = identical initial weights!
    
    # Verify identical initialization
    sgd_params = [p.clone() for p in sgd_model.parameters()]
    sgld_params = [p.clone() for p in sgld_model.parameters()]
    weights_identical = all(torch.equal(sgd_p, sgld_p) for sgd_p, sgld_p in zip(sgd_params, sgld_params))
    print(f"✓ Weights identical: {weights_identical}")
    
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=LR)
    sgd_history, sgd_time = train_model(
        sgd_model, train_loader, val_loader, sgd_optimizer, NUM_EPOCHS, DEVICE, "SGD"
    )
    
    sgld_optimizer = SGLD(
        sgld_model.parameters(), 
        lr=LR, 
        temperature=TEMPERATURE, 
        noise_decay=NOISE_DECAY,
        lr_decay=0.001,  # Very small LR decay to keep it almost constant
        min_lr=LR * 0.5  # Allow some decay but not too much
    )
    sgld_history, sgld_time = train_model(
        sgld_model, train_loader, val_loader, sgld_optimizer, NUM_EPOCHS, DEVICE, "SGLD"
    )
    
    # Final evaluation
    print("\nFinal Test Set Evaluation:")
    sgd_test_acc = evaluate_model(sgd_model, test_loader, DEVICE)
    sgld_test_acc = evaluate_model(sgld_model, test_loader, DEVICE)
    
    print(f"SGD Test Accuracy:  {sgd_test_acc:.1f}%")
    print(f"SGLD Test Accuracy: {sgld_test_acc:.1f}%")
    print(f"Difference: {sgld_test_acc - sgd_test_acc:.1f}%")
    print()
    
    print(f"SGD Training Time:  {sgd_time:.1f}s")
    print(f"SGLD Training Time: {sgld_time:.1f}s")
    print()
    
    # Plot results
    plot_comparison(sgd_history, sgld_history)
    
    # Winner
    if sgld_test_acc > sgd_test_acc:
        print(f"🏆 SGLD wins by {sgld_test_acc - sgd_test_acc:.1f}%")
    elif sgd_test_acc > sgld_test_acc:
        print(f"🏆 SGD wins by {sgd_test_acc - sgld_test_acc:.1f}%")
    else:
        print("🤝 It's a tie!")
    
    return {
        'sgd_history': sgd_history,
        'sgld_history': sgld_history,
        'sgd_test_acc': sgd_test_acc,
        'sgld_test_acc': sgld_test_acc,
        'sgd_time': sgd_time,
        'sgld_time': sgld_time
    }


if __name__ == "__main__":
    results = run_experiment()

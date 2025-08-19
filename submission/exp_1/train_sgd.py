"""
SGD Training Script for CIFAR-10

Run this in parallel with train_sgld.py
"""

import sys
sys.path.append('..')  # To access shared sgld.py

import torch
import torch.nn as nn
import pickle
import time

from model import get_cifar10_model
from cifar10_loader import load_cifar10_data

# Check for tqdm
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def train_sgd():
    """Train with SGD optimizer."""
    print("SGD Training Script")
    print("=" * 30)
    
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    LR = 0.01
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Optimizer: SGD (no momentum, no weight decay)")
    print()
    
    # Load data
    print("Loading CIFAR-10 data...")
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()
    
    # Model and optimizer
    model = get_cifar10_model().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)  # No momentum, no weight decay
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        if HAVE_TQDM:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
        else:
            pbar = train_loader
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}: ', end='', flush=True)
        
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
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
                data, target = data.to(DEVICE), target.to(DEVICE)
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
        
        print(f'SGD Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    training_time = time.time() - start_time
    
    print(f"\nSGD Final Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.1f}s")
    
    # Save results
    results = {
        'history': history,
        'test_acc': test_acc,
        'training_time': training_time,
        'model_state': model.state_dict(),
        'config': {
            'optimizer': 'SGD',
            'lr': LR,
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE
        }
    }
    
    with open('sgd_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved to sgd_results.pkl")
    
    return results


if __name__ == "__main__":
    train_sgd()

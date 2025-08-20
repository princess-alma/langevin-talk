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
import numpy as np

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
    """Train a model and return training history + saved checkpoints for ensembling."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    saved_models = []  # Store model checkpoints for ensembling
    
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
        
        # Save best models for ensembling (after epoch 500)
        if epoch + 1 >= 500:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {k: v.clone().cpu() for k, v in model.state_dict().items()},
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss / len(val_loader)
            }
            
            # Add to saved models and keep only top 5 by validation loss (lower is better)
            saved_models.append(checkpoint)
            saved_models.sort(key=lambda x: x['val_loss'])  # Sort by loss, ascending (lower is better)
            
            if len(saved_models) > 5:
                saved_models = saved_models[:5]
                
            # Print when we save a new best model or update the top 5
            if len(saved_models) <= 5 and (epoch + 1) % 100 == 0:
                max_val_loss = max(m['val_loss'] for m in saved_models) if saved_models else 0
                print(f'  💾 Top 5 models: {len(saved_models)}/5, max val_loss: {max_val_loss:.4f}')
    
    training_time = time.time() - start_time
    print(f'{optimizer_name} Training Time: {training_time:.1f}s')
    
    if saved_models:
        best_val_loss = min(m['val_loss'] for m in saved_models)  # Lower loss is better
        worst_val_loss = max(m['val_loss'] for m in saved_models)
        avg_val_loss = sum(m['val_loss'] for m in saved_models) / len(saved_models)
        print(f'{optimizer_name} Saved {len(saved_models)} best models (val loss: {best_val_loss:.4f}-{worst_val_loss:.4f}, avg: {avg_val_loss:.4f})')
    
    return history, training_time, saved_models


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


def get_shared_weights():
    """Generate shared initial weights for fair comparison."""
    model = get_moons_model()
    # Use a random seed for this run, but same for both SGD and SGLD
    torch.manual_seed(random.randint(1, 10000))
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    return model.state_dict()


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


def evaluate_ensemble(saved_models, test_loader, device):
    """Evaluate ensemble performance by averaging predictions from saved checkpoints."""
    ensemble_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Collect predictions from all saved models
            ensemble_logits = torch.zeros(data.size(0), 2).to(device)
            
            for checkpoint in saved_models:
                # Create model and load checkpoint
                model = get_moons_model().to(device)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                
                # Add this model's prediction to the ensemble
                outputs = model(data)
                ensemble_logits += outputs
            
            # Average the predictions
            ensemble_logits /= len(saved_models)
            _, predicted = torch.max(ensemble_logits, 1)
            
            total += target.size(0)
            ensemble_correct += (predicted == target).sum().item()
    
    return 100. * ensemble_correct / total


def run_experiment():
    """Run the complete moons experiment with ensemble evaluation."""
    print("Moons Dataset Experiment: SGD vs SGLD with Ensemble Evaluation")
    print("=" * 64)
    
    # Optimal configuration based on our experiments
    BATCH_SIZE = 32
    NUM_EPOCHS = 1500  # Extended training for clear differences
    LR = 0.01  # Higher LR works well for small networks
    TEMPERATURE = 0.05  # Moderate initial temperature
    NOISE_DECAY = 0.9996  # Much slower decay to maintain exploration
    DEVICE = torch.device('cpu')
    
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
    
    # Generate shared weights for identical initialization
    shared_weights = get_shared_weights()
    print("Generated shared weights for identical initialization")
    
    # Train with SGD
    model_sgd = get_moons_model()
    model_sgd.load_state_dict(shared_weights)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=LR)
    
    sgd_history, sgd_time, sgd_models = train_model(
        model_sgd, train_loader, val_loader, optimizer_sgd, NUM_EPOCHS, DEVICE, "SGD"
    )
    
    # Train with SGLD
    model_sgld = get_moons_model()
    model_sgld.load_state_dict(shared_weights)
    optimizer_sgld = SGLD(
        model_sgld.parameters(), 
        lr=LR, 
        temperature=TEMPERATURE,
        noise_decay=NOISE_DECAY
    )
    
    sgld_history, sgld_time, sgld_models = train_model(
        model_sgld, train_loader, val_loader, optimizer_sgld, NUM_EPOCHS, DEVICE, "SGLD"
    )
    
    # Individual model evaluation
    sgd_test_acc = evaluate_model(model_sgd, test_loader, DEVICE)
    sgld_test_acc = evaluate_model(model_sgld, test_loader, DEVICE)
    
    # Ensemble evaluation
    print("\n" + "="*60)
    print("ENSEMBLE PERFORMANCE COMPARISON")
    print("="*60)
    
    sgd_ensemble_acc = evaluate_ensemble(sgd_models, test_loader, DEVICE)
    sgld_ensemble_acc = evaluate_ensemble(sgld_models, test_loader, DEVICE)
    
    print(f"SGD Ensemble Test Accuracy:  {sgd_ensemble_acc:.1f}% ({len(sgd_models)} models)")
    print(f"SGLD Ensemble Test Accuracy: {sgld_ensemble_acc:.1f}% ({len(sgld_models)} models)")
    print(f"Ensemble Difference: {sgld_ensemble_acc - sgd_ensemble_acc:.1f}%")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"SGD:  Individual {sgd_test_acc:.1f}% → Ensemble {sgd_ensemble_acc:.1f}% (Δ{sgd_ensemble_acc - sgd_test_acc:+.1f}%)")
    print(f"SGLD: Individual {sgld_test_acc:.1f}% → Ensemble {sgld_ensemble_acc:.1f}% (Δ{sgld_ensemble_acc - sgld_test_acc:+.1f}%)")
    print(f"Training Times: SGD {sgd_time:.1f}s, SGLD {sgld_time:.1f}s")
    
    if sgld_ensemble_acc > sgd_ensemble_acc:
        print(f"\n🏆 SGLD ensemble wins by {sgld_ensemble_acc - sgd_ensemble_acc:.1f}%!")
    elif sgd_ensemble_acc > sgld_ensemble_acc:
        print(f"\n🏆 SGD ensemble wins by {sgd_ensemble_acc - sgld_ensemble_acc:.1f}%!")
    else:
        print("\n🤝 Ensemble tie!")
    
    # Plot comparison
    plot_comparison(sgd_history, sgld_history)
    
    return {
        'sgd_history': sgd_history,
        'sgld_history': sgld_history,
        'sgd_test_acc': sgd_test_acc,
        'sgld_test_acc': sgld_test_acc,
        'sgd_ensemble_acc': sgd_ensemble_acc,
        'sgld_ensemble_acc': sgld_ensemble_acc,
        'sgd_time': sgd_time,
        'sgld_time': sgld_time,
        'sgd_models': len(sgd_models),
        'sgld_models': len(sgld_models)
    }


def run_multiple_experiments(n_trials=100):
    """Run the experiment n times and collect statistics."""
    print(f"Running {n_trials} trials of SGD vs SGLD comparison...")
    print("=" * 60)
    
    # Track wins for individual models and ensembles
    sgd_individual_wins = 0
    sgld_individual_wins = 0
    individual_ties = 0
    
    sgd_ensemble_wins = 0
    sgld_ensemble_wins = 0
    ensemble_ties = 0
    
    # Track detailed results
    individual_differences = []  # sgld_acc - sgd_acc
    ensemble_differences = []    # sgld_ensemble - sgd_ensemble
    
    sgd_individual_accs = []
    sgld_individual_accs = []
    sgd_ensemble_accs = []
    sgld_ensemble_accs = []
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 30)
        
        # Run single experiment (suppress detailed output)
        results = run_single_experiment_quiet()
        
        # Extract results
        sgd_acc = results['sgd_test_acc']
        sgld_acc = results['sgld_test_acc']
        sgd_ens = results['sgd_ensemble_acc']
        sgld_ens = results['sgld_ensemble_acc']
        
        # Store results
        sgd_individual_accs.append(sgd_acc)
        sgld_individual_accs.append(sgld_acc)
        sgd_ensemble_accs.append(sgd_ens)
        sgld_ensemble_accs.append(sgld_ens)
        
        individual_diff = sgld_acc - sgd_acc
        ensemble_diff = sgld_ens - sgd_ens
        individual_differences.append(individual_diff)
        ensemble_differences.append(ensemble_diff)
        
        # Count wins for individual models
        if sgld_acc > sgd_acc:
            sgld_individual_wins += 1
        elif sgd_acc > sgld_acc:
            sgd_individual_wins += 1
        else:
            individual_ties += 1
            
        # Count wins for ensembles
        if sgld_ens > sgd_ens:
            sgld_ensemble_wins += 1
        elif sgd_ens > sgld_ens:
            sgd_ensemble_wins += 1
        else:
            ensemble_ties += 1
        
        # Quick summary for this trial
        print(f"Individual: SGD {sgd_acc:.1f}% vs SGLD {sgld_acc:.1f}% (diff: {individual_diff:+.1f}%)")
        print(f"Ensemble:   SGD {sgd_ens:.1f}% vs SGLD {sgld_ens:.1f}% (diff: {ensemble_diff:+.1f}%)")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS ACROSS ALL TRIALS")
    print("=" * 60)
    
    print(f"\nINDIVIDUAL MODEL WINS:")
    print(f"  SGD wins:    {sgd_individual_wins}/{n_trials} ({100*sgd_individual_wins/n_trials:.1f}%)")
    print(f"  SGLD wins:   {sgld_individual_wins}/{n_trials} ({100*sgld_individual_wins/n_trials:.1f}%)")
    print(f"  Ties:        {individual_ties}/{n_trials} ({100*individual_ties/n_trials:.1f}%)")
    
    print(f"\nENSEMBLE MODEL WINS:")
    print(f"  SGD wins:    {sgd_ensemble_wins}/{n_trials} ({100*sgd_ensemble_wins/n_trials:.1f}%)")
    print(f"  SGLD wins:   {sgld_ensemble_wins}/{n_trials} ({100*sgld_ensemble_wins/n_trials:.1f}%)")
    print(f"  Ties:        {ensemble_ties}/{n_trials} ({100*ensemble_ties/n_trials:.1f}%)")
    
    # Statistical summaries
    import numpy as np
    
    print(f"\nPERFORMANCE STATISTICS:")
    print(f"Individual Models:")
    print(f"  SGD:  {np.mean(sgd_individual_accs):.2f}% ± {np.std(sgd_individual_accs):.2f}%")
    print(f"  SGLD: {np.mean(sgld_individual_accs):.2f}% ± {np.std(sgld_individual_accs):.2f}%")
    print(f"  Difference (SGLD - SGD): {np.mean(individual_differences):.2f}% ± {np.std(individual_differences):.2f}%")
    
    print(f"\nEnsemble Models:")
    print(f"  SGD:  {np.mean(sgd_ensemble_accs):.2f}% ± {np.std(sgd_ensemble_accs):.2f}%")
    print(f"  SGLD: {np.mean(sgld_ensemble_accs):.2f}% ± {np.std(sgld_ensemble_accs):.2f}%")
    print(f"  Difference (SGLD - SGD): {np.mean(ensemble_differences):.2f}% ± {np.std(ensemble_differences):.2f}%")
    
    # Winner determination
    if sgld_ensemble_wins > sgd_ensemble_wins:
        print(f"\n🏆 OVERALL WINNER: SGLD (wins {sgld_ensemble_wins}/{n_trials} ensemble comparisons)")
    elif sgd_ensemble_wins > sgld_ensemble_wins:
        print(f"\n🏆 OVERALL WINNER: SGD (wins {sgd_ensemble_wins}/{n_trials} ensemble comparisons)")
    else:
        print(f"\n🤝 OVERALL TIE: SGD and SGLD each win {sgd_ensemble_wins}/{n_trials} times")
    
    return {
        'sgd_individual_wins': sgd_individual_wins,
        'sgld_individual_wins': sgld_individual_wins,
        'sgd_ensemble_wins': sgd_ensemble_wins,
        'sgld_ensemble_wins': sgld_ensemble_wins,
        'individual_differences': individual_differences,
        'ensemble_differences': ensemble_differences,
        'sgd_individual_accs': sgd_individual_accs,
        'sgld_individual_accs': sgld_individual_accs,
        'sgd_ensemble_accs': sgd_ensemble_accs,
        'sgld_ensemble_accs': sgld_ensemble_accs
    }


def run_single_experiment_quiet():
    """Run a single experiment with minimal output."""
    # Use the SAME hyperparameters as the main experiment
    BATCH_SIZE = 32
    NUM_EPOCHS = 1500
    LR = 0.01
    TEMPERATURE = 0.01
    NOISE_DECAY = 0.9996
    DEVICE = torch.device('cpu')
    
    # Load data
    train_loader, val_loader, test_loader = load_moons_data(
        n_samples=1000, noise=0.1, batch_size=BATCH_SIZE
    )
    
    # Generate shared weights for identical initialization
    shared_weights = get_shared_weights()
    
    # Train with SGD (quiet)
    model_sgd = get_moons_model()
    model_sgd.load_state_dict(shared_weights)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=LR)
    
    sgd_history, sgd_time, sgd_models = train_model_quiet(
        model_sgd, train_loader, val_loader, optimizer_sgd, NUM_EPOCHS, DEVICE, "SGD"
    )
    
    # Train with SGLD (quiet)
    model_sgld = get_moons_model()
    model_sgld.load_state_dict(shared_weights)
    optimizer_sgld = SGLD(
        model_sgld.parameters(), 
        lr=LR, 
        temperature=TEMPERATURE,
        noise_decay=NOISE_DECAY
    )
    
    sgld_history, sgld_time, sgld_models = train_model_quiet(
        model_sgld, train_loader, val_loader, optimizer_sgld, NUM_EPOCHS, DEVICE, "SGLD"
    )
    
    # Evaluate
    sgd_test_acc = evaluate_model(model_sgd, test_loader, DEVICE)
    sgld_test_acc = evaluate_model(model_sgld, test_loader, DEVICE)
    
    sgd_ensemble_acc = evaluate_ensemble(sgd_models, test_loader, DEVICE)
    sgld_ensemble_acc = evaluate_ensemble(sgld_models, test_loader, DEVICE)
    
    return {
        'sgd_history': sgd_history,
        'sgld_history': sgld_history,
        'sgd_test_acc': sgd_test_acc,
        'sgld_test_acc': sgld_test_acc,
        'sgd_ensemble_acc': sgd_ensemble_acc,
        'sgld_ensemble_acc': sgld_ensemble_acc,
        'sgd_time': sgd_time,
        'sgld_time': sgld_time,
        'sgd_models': len(sgd_models),
        'sgld_models': len(sgld_models)
    }


def train_model_quiet(model, train_loader, val_loader, optimizer, num_epochs, device, optimizer_name):
    """Train a model with minimal output (for multiple trials)."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    saved_models = []
    
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
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
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
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Record history
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Save best models for ensembling (after epoch 500)
        if epoch + 1 >= 500:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {k: v.clone().cpu() for k, v in model.state_dict().items()},
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss / len(val_loader)
            }
            
            saved_models.append(checkpoint)
            saved_models.sort(key=lambda x: x['val_loss'])
            
            if len(saved_models) > 5:
                saved_models = saved_models[:5]
    
    training_time = time.time() - start_time
    return history, training_time, saved_models


if __name__ == "__main__":
    # Run a single SGD vs SGLD comparison
    results = run_experiment()

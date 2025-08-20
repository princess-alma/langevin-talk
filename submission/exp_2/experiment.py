"""
Moons Experiment: SGD vs SGLD

Clean experiment comparing SGD vs SGLD on the two moons dataset.
This demonstrates SGLD's exploration advantages on complex loss landscapes.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

from model import get_moons_model
from sgld import SGLD
from moons_loader import load_moons_data
import random

# Experimental Configuration
NUM_EXPERIMENTS = 10
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LR = 0.01
TEMPERATURE = 0.002
TEMPERATURE_DECAY = 0.999
DEVICE = torch.device('cpu')


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, optimizer_name):
    """Train a model and return training history + saved checkpoints for ensembling."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    saved_models = []  # Store model checkpoints for ensembling
    
    print(f"\nTraining with {optimizer_name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Set the current epoch for SGLD temperature decay
        if hasattr(optimizer, 'set_epoch'):
            optimizer.set_epoch(epoch)
            
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
        
        # Save model checkpoints every 100 epochs after epoch 500 (not based on best val loss)
        if epoch + 1 >= 500 and (epoch + 1) % 100 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {k: v.clone().cpu() for k, v in model.state_dict().items()},
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss / len(val_loader)
            }
            
            # Add to saved models - save every 100 epochs, not based on performance
            saved_models.append(checkpoint)
            
            print(f'  💾 Saved checkpoint at epoch {epoch + 1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%, Val Loss: {val_loss / len(val_loader):.4f}')
    
    training_time = time.time() - start_time
    print(f'{optimizer_name} Training Time: {training_time:.1f}s')
    
    if saved_models:
        latest_epoch = max(m['epoch'] for m in saved_models)
        earliest_epoch = min(m['epoch'] for m in saved_models)
        avg_val_loss = sum(m['val_loss'] for m in saved_models) / len(saved_models)
        print(f'{optimizer_name} Saved {len(saved_models)} checkpoint models (epochs {earliest_epoch}-{latest_epoch}, avg val loss: {avg_val_loss:.4f})')
    
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


def evaluate_ensemble(saved_models, test_loader, device):
    """Evaluate ensemble performance by averaging predictions from saved checkpoints."""
    ensemble_correct = 0
    total = 0
    
    # Load all models once, outside the batch loop (efficiency fix)
    models = []
    for checkpoint in saved_models:
        model = get_moons_model().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models.append(model)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Collect predictions from all saved models
            ensemble_logits = torch.zeros(data.size(0), 2).to(device)
            
            for model in models:
                # Add this model's prediction to the ensemble
                outputs = model(data)
                ensemble_logits += outputs
            
            # Average the predictions
            ensemble_logits /= len(models)
            _, predicted = torch.max(ensemble_logits, 1)
            
            total += target.size(0)
            ensemble_correct += (predicted == target).sum().item()
    
    return 100. * ensemble_correct / total


def run_experiment():
    """Run the complete moons experiment with ensemble evaluation."""
    print("Moons Dataset Experiment: SGD vs SGLD with Ensemble Evaluation")
    print("=" * 64)
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"SGLD Temperature: {TEMPERATURE}")
    print(f"SGLD Temperature Decay: {TEMPERATURE_DECAY}")
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
        temperature_decay=TEMPERATURE_DECAY
    )
    
    sgld_history, sgld_time, sgld_models = train_model(
        model_sgld, train_loader, val_loader, optimizer_sgld, NUM_EPOCHS, DEVICE, "SGLD"
    )
    
    # Individual model evaluation
    sgd_test_acc = evaluate_model(model_sgd, test_loader, DEVICE)
    sgld_test_acc = evaluate_model(model_sgld, test_loader, DEVICE)
    
    # Ensemble evaluation
    print("\n" + "="*60)
    print("ENSEMBLE PERFORMANCE COMPARISON (Regular Checkpoints)")
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


def setup_ax(ax, title, xlabel='Epoch', ylabel='Value'):
    """Helper to format a subplot (without legend)."""
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_multiple_experiments(all_results, num_experiments):
    """Plot learning curves from multiple experiment runs - create two plots: all runs + averages only."""
    
    # Plot 1: All individual runs with averages
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors_sgd = plt.cm.Blues(np.linspace(0.4, 0.8, num_experiments))
    colors_sgld = plt.cm.Reds(np.linspace(0.4, 0.8, num_experiments))
    
    # Plot individual runs
    for i, result in enumerate(all_results):
        sgd_history = result['sgd_history']
        sgld_history = result['sgld_history']
        epochs = range(1, len(sgd_history['train_loss']) + 1)
        
        alpha = 0.6 if num_experiments > 1 else 1.0
        label_sgd = 'SGD Individual Runs' if i == 0 else None
        label_sgld = 'SGLD Individual Runs' if i == 0 else None
        
        # Training Loss
        ax1.plot(epochs, sgd_history['train_loss'], color=colors_sgd[i], 
                alpha=alpha, linewidth=1.2, label=label_sgd)
        ax1.plot(epochs, sgld_history['train_loss'], color=colors_sgld[i], 
                alpha=alpha, linewidth=1.2, label=label_sgld)
        
        # Validation Loss
        ax2.plot(epochs, sgd_history['val_loss'], color=colors_sgd[i], alpha=alpha, linewidth=1.2)
        ax2.plot(epochs, sgld_history['val_loss'], color=colors_sgld[i], alpha=alpha, linewidth=1.2)
        
        # Training Accuracy
        ax3.plot(epochs, sgd_history['train_acc'], color=colors_sgd[i], alpha=alpha, linewidth=1.2)
        ax3.plot(epochs, sgld_history['train_acc'], color=colors_sgld[i], alpha=alpha, linewidth=1.2)
        
        # Validation Accuracy
        ax4.plot(epochs, sgd_history['val_acc'], color=colors_sgd[i], alpha=alpha, linewidth=1.2)
        ax4.plot(epochs, sgld_history['val_acc'], color=colors_sgld[i], alpha=alpha, linewidth=1.2)
    
    # Add average lines if multiple runs
    if num_experiments > 1:
        # Calculate averages
        avg_sgd_train_loss = np.mean([r['sgd_history']['train_loss'] for r in all_results], axis=0)
        avg_sgld_train_loss = np.mean([r['sgld_history']['train_loss'] for r in all_results], axis=0)
        avg_sgd_val_loss = np.mean([r['sgd_history']['val_loss'] for r in all_results], axis=0)
        avg_sgld_val_loss = np.mean([r['sgld_history']['val_loss'] for r in all_results], axis=0)
        avg_sgd_train_acc = np.mean([r['sgd_history']['train_acc'] for r in all_results], axis=0)
        avg_sgld_train_acc = np.mean([r['sgld_history']['train_acc'] for r in all_results], axis=0)
        avg_sgd_val_acc = np.mean([r['sgd_history']['val_acc'] for r in all_results], axis=0)
        avg_sgld_val_acc = np.mean([r['sgld_history']['val_acc'] for r in all_results], axis=0)
        
        epochs = range(1, len(avg_sgd_train_loss) + 1)
        
        # Plot averages with thicker lines
        ax1.plot(epochs, avg_sgd_train_loss, 'b-', linewidth=3, label='SGD Average')
        ax1.plot(epochs, avg_sgld_train_loss, 'r-', linewidth=3, label='SGLD Average')
        
        ax2.plot(epochs, avg_sgd_val_loss, 'b-', linewidth=3, label='SGD Average')
        ax2.plot(epochs, avg_sgld_val_loss, 'r-', linewidth=3, label='SGLD Average')
        
        ax3.plot(epochs, avg_sgd_train_acc, 'b-', linewidth=3, label='SGD Average')
        ax3.plot(epochs, avg_sgld_train_acc, 'r-', linewidth=3, label='SGLD Average')
        
        ax4.plot(epochs, avg_sgd_val_acc, 'b-', linewidth=3, label='SGD Average')
        ax4.plot(epochs, avg_sgld_val_acc, 'r-', linewidth=3, label='SGLD Average')
    
    # Formatting for Plot 1
    setup_ax(ax1, f'Training Loss - All Runs ({num_experiments} experiments)', ylabel='Loss')
    setup_ax(ax2, f'Validation Loss - All Runs ({num_experiments} experiments)', ylabel='Loss')
    setup_ax(ax3, f'Training Accuracy - All Runs ({num_experiments} experiments)', ylabel='Accuracy (%)')
    setup_ax(ax4, f'Validation Accuracy - All Runs ({num_experiments} experiments)', ylabel='Accuracy (%)')
    
    # Add a single shared legend at the top of the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the legend
    filename1 = f'moons_all_runs_{num_experiments}experiments.png'
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    print(f"📊 All runs plot saved as '{filename1}'")
    plt.show()
    
    # Plot 2: Averages only
    if num_experiments > 1:
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(avg_sgd_train_loss) + 1)
        
        # Plot only averages with confidence intervals/standard deviation
        sgd_train_loss_std = np.std([r['sgd_history']['train_loss'] for r in all_results], axis=0)
        sgld_train_loss_std = np.std([r['sgld_history']['train_loss'] for r in all_results], axis=0)
        sgd_val_loss_std = np.std([r['sgd_history']['val_loss'] for r in all_results], axis=0)
        sgld_val_loss_std = np.std([r['sgld_history']['val_loss'] for r in all_results], axis=0)
        sgd_train_acc_std = np.std([r['sgd_history']['train_acc'] for r in all_results], axis=0)
        sgld_train_acc_std = np.std([r['sgld_history']['train_acc'] for r in all_results], axis=0)
        sgd_val_acc_std = np.std([r['sgd_history']['val_acc'] for r in all_results], axis=0)
        sgld_val_acc_std = np.std([r['sgld_history']['val_acc'] for r in all_results], axis=0)
        
        # Training Loss with confidence bands (capped at 0)
        ax1.plot(epochs, avg_sgd_train_loss, 'b-', linewidth=3, label='SGD')
        ax1.fill_between(epochs, 
                         np.maximum(0, avg_sgd_train_loss - sgd_train_loss_std), 
                         avg_sgd_train_loss + sgd_train_loss_std, alpha=0.2, color='blue')
        ax1.plot(epochs, avg_sgld_train_loss, 'r-', linewidth=3, label='SGLD')
        ax1.fill_between(epochs, 
                         np.maximum(0, avg_sgld_train_loss - sgld_train_loss_std), 
                         avg_sgld_train_loss + sgld_train_loss_std, alpha=0.2, color='red')
        
        # Validation Loss with confidence bands (capped at 0)
        ax2.plot(epochs, avg_sgd_val_loss, 'b-', linewidth=3, label='SGD')
        ax2.fill_between(epochs, 
                         np.maximum(0, avg_sgd_val_loss - sgd_val_loss_std), 
                         avg_sgd_val_loss + sgd_val_loss_std, alpha=0.2, color='blue')
        ax2.plot(epochs, avg_sgld_val_loss, 'r-', linewidth=3, label='SGLD')
        ax2.fill_between(epochs, 
                         np.maximum(0, avg_sgld_val_loss - sgld_val_loss_std), 
                         avg_sgld_val_loss + sgld_val_loss_std, alpha=0.2, color='red')
        
        # Training Accuracy with confidence bands (capped at 0% and 100%)
        ax3.plot(epochs, avg_sgd_train_acc, 'b-', linewidth=3, label='SGD')
        ax3.fill_between(epochs, 
                         np.maximum(0, avg_sgd_train_acc - sgd_train_acc_std), 
                         np.minimum(100, avg_sgd_train_acc + sgd_train_acc_std), alpha=0.2, color='blue')
        ax3.plot(epochs, avg_sgld_train_acc, 'r-', linewidth=3, label='SGLD')
        ax3.fill_between(epochs, 
                         np.maximum(0, avg_sgld_train_acc - sgld_train_acc_std), 
                         np.minimum(100, avg_sgld_train_acc + sgld_train_acc_std), alpha=0.2, color='red')
        
        # Validation Accuracy with confidence bands (capped at 0% and 100%)
        ax4.plot(epochs, avg_sgd_val_acc, 'b-', linewidth=3, label='SGD')
        ax4.fill_between(epochs, 
                         np.maximum(0, avg_sgd_val_acc - sgd_val_acc_std), 
                         np.minimum(100, avg_sgd_val_acc + sgd_val_acc_std), alpha=0.2, color='blue')
        ax4.plot(epochs, avg_sgld_val_acc, 'r-', linewidth=3, label='SGLD')
        ax4.fill_between(epochs, 
                         np.maximum(0, avg_sgld_val_acc - sgld_val_acc_std), 
                         np.minimum(100, avg_sgld_val_acc + sgld_val_acc_std), alpha=0.2, color='red')
        
        # Formatting for Plot 2
        setup_ax(ax1, f'Training Loss - Averages ({num_experiments} experiments)', ylabel='Loss')
        setup_ax(ax2, f'Validation Loss - Averages ({num_experiments} experiments)', ylabel='Loss')
        setup_ax(ax3, f'Training Accuracy - Averages ({num_experiments} experiments)', ylabel='Accuracy (%)')
        setup_ax(ax4, f'Validation Accuracy - Averages ({num_experiments} experiments)', ylabel='Accuracy (%)')
        
        # Add a single shared legend at the top of the figure
        handles, labels = ax1.get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for the legend
        filename2 = f'moons_averages_{num_experiments}experiments.png'
        plt.savefig(filename2, dpi=150, bbox_inches='tight')
        print(f"📊 Averages plot saved as '{filename2}'")
        plt.show()
    else:
        print("📊 Single experiment - only one plot generated")


def print_summary_statistics(all_results, num_experiments):
    """Print summary statistics across all experiments."""
    print("\n" + "="*70)
    print(f"SUMMARY STATISTICS ACROSS {num_experiments} EXPERIMENTS")
    print("="*70)
    
    # Individual model performance
    sgd_test_accs = [r['sgd_test_acc'] for r in all_results]
    sgld_test_accs = [r['sgld_test_acc'] for r in all_results]
    
    # Ensemble performance
    sgd_ensemble_accs = [r['sgd_ensemble_acc'] for r in all_results]
    sgld_ensemble_accs = [r['sgld_ensemble_acc'] for r in all_results]
    
    # Training times
    sgd_times = [r['sgd_time'] for r in all_results]
    sgld_times = [r['sgld_time'] for r in all_results]
    
    print("\nIndividual Model Performance:")
    print(f"SGD Test Accuracy:  {np.mean(sgd_test_accs):.1f}% ± {np.std(sgd_test_accs):.1f}%")
    print(f"SGLD Test Accuracy: {np.mean(sgld_test_accs):.1f}% ± {np.std(sgld_test_accs):.1f}%")
    
    print("\nEnsemble Performance:")
    print(f"SGD Ensemble Accuracy:  {np.mean(sgd_ensemble_accs):.1f}% ± {np.std(sgd_ensemble_accs):.1f}%")
    print(f"SGLD Ensemble Accuracy: {np.mean(sgld_ensemble_accs):.1f}% ± {np.std(sgld_ensemble_accs):.1f}%")
    
    print("\nTraining Times:")
    print(f"SGD Training Time:  {np.mean(sgd_times):.1f}s ± {np.std(sgd_times):.1f}s")
    print(f"SGLD Training Time: {np.mean(sgld_times):.1f}s ± {np.std(sgld_times):.1f}s")
    
    # Overall winner announcement (if multiple runs)
    if num_experiments > 1:
        # Calculate performance improvements for winner determination
        ensemble_improvements = [sgld - sgd for sgld, sgd in zip(sgld_ensemble_accs, sgd_ensemble_accs)]
        
        avg_ensemble_improvement = np.mean(ensemble_improvements)
        if avg_ensemble_improvement > 0.5:  # More than 0.5% improvement
            print(f"\n🏆 SGLD ensemble wins overall by {avg_ensemble_improvement:.1f}%!")
        elif avg_ensemble_improvement < -0.5:  # More than 0.5% worse
            print(f"\n🏆 SGD ensemble wins overall by {-avg_ensemble_improvement:.1f}%!")
        else:
            print(f"\n🤝 Overall ensemble performance is essentially tied!")
    else:
        # Single experiment case
        result = all_results[0]
        if result['sgld_ensemble_acc'] > result['sgd_ensemble_acc'] + 0.5:
            print(f"\n🏆 SGLD ensemble wins by {result['sgld_ensemble_acc'] - result['sgd_ensemble_acc']:.1f}%!")
        elif result['sgd_ensemble_acc'] > result['sgld_ensemble_acc'] + 0.5:
            print(f"\n🏆 SGD ensemble wins by {result['sgd_ensemble_acc'] - result['sgld_ensemble_acc']:.1f}%!")
        else:
            print(f"\n🤝 Ensemble performance is essentially tied!")


if __name__ == "__main__":    
    print(f"Running {NUM_EXPERIMENTS} independent experiments...")
    print("=" * 60)
    
    # Store results from all experiments
    all_results = []
    
    for experiment_idx in range(NUM_EXPERIMENTS):
        print(f"\n🔄 EXPERIMENT {experiment_idx + 1}/{NUM_EXPERIMENTS}")
        print("-" * 40)
        
        # Run experiment and collect results
        results = run_experiment()
        all_results.append(results)
        
        print(f"✅ Experiment {experiment_idx + 1} completed")
        
        # Brief summary for this run
        print(f"   SGD: Individual {results['sgd_test_acc']:.1f}% → Ensemble {results['sgd_ensemble_acc']:.1f}%")
        print(f"   SGLD: Individual {results['sgld_test_acc']:.1f}% → Ensemble {results['sgld_ensemble_acc']:.1f}%")
    
    # Print comprehensive summary statistics
    print_summary_statistics(all_results, NUM_EXPERIMENTS)
    
    # Visualize all learning curves
    plot_multiple_experiments(all_results, NUM_EXPERIMENTS)

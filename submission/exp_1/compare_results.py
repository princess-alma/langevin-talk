"""
Compare Results Script

Run this after both train_sgd.py and train_sgld.py have completed.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load results from both training scripts."""
    try:
        with open('sgd_results.pkl', 'rb') as f:
            sgd_results = pickle.load(f)
        print("✓ Loaded SGD results")
    except FileNotFoundError:
        print("✗ SGD results not found. Run train_sgd.py first.")
        return None, None
    
    try:
        with open('sgld_results.pkl', 'rb') as f:
            sgld_results = pickle.load(f)
        print("✓ Loaded SGLD results")
    except FileNotFoundError:
        print("✗ SGLD results not found. Run train_sgld.py first.")
        return sgd_results, None
    
    return sgd_results, sgld_results


def compare_results(sgd_results, sgld_results):
    """Compare and visualize results."""
    if sgd_results is None or sgld_results is None:
        print("Missing results files. Cannot compare.")
        return
    
    print("\nComparison Results:")
    print("=" * 50)
    
    # Test accuracies
    print(f"SGD Test Accuracy:  {sgd_results['test_acc']:.2f}%")
    print(f"SGLD Test Accuracy: {sgld_results['test_acc']:.2f}%")
    print(f"Difference: {sgld_results['test_acc'] - sgd_results['test_acc']:.2f}%")
    print()
    
    # Training times
    print(f"SGD Training Time:  {sgd_results['training_time']:.1f}s")
    print(f"SGLD Training Time: {sgld_results['training_time']:.1f}s")
    print(f"Time Difference: {sgld_results['training_time'] - sgd_results['training_time']:.1f}s")
    print()
    
    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    sgd_hist = sgd_results['history']
    sgld_hist = sgld_results['history']
    epochs = range(1, len(sgd_hist['train_loss']) + 1)
    
    # Training loss
    ax1.plot(epochs, sgd_hist['train_loss'], 'b-', label='SGD', linewidth=2)
    ax1.plot(epochs, sgld_hist['train_loss'], 'r-', label='SGLD', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation loss
    ax2.plot(epochs, sgd_hist['val_loss'], 'b-', label='SGD', linewidth=2)
    ax2.plot(epochs, sgld_hist['val_loss'], 'r-', label='SGLD', linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training accuracy
    ax3.plot(epochs, sgd_hist['train_acc'], 'b-', label='SGD', linewidth=2)
    ax3.plot(epochs, sgld_hist['train_acc'], 'r-', label='SGLD', linewidth=2)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax4.plot(epochs, sgd_hist['val_acc'], 'b-', label='SGD', linewidth=2)
    ax4.plot(epochs, sgld_hist['val_acc'], 'r-', label='SGLD', linewidth=2)
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sgd_vs_sgld_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Comparison plot saved as 'sgd_vs_sgld_comparison.png'")
    plt.show()
    
    # Final summary
    print("\nFinal Summary:")
    print("-" * 30)
    if sgld_results['test_acc'] > sgd_results['test_acc']:
        print(f"🏆 SGLD wins by {sgld_results['test_acc'] - sgd_results['test_acc']:.2f}%")
    elif sgd_results['test_acc'] > sgld_results['test_acc']:
        print(f"🏆 SGD wins by {sgd_results['test_acc'] - sgld_results['test_acc']:.2f}%")
    else:
        print("🤝 It's a tie!")


def main():
    """Main comparison function."""
    print("SGD vs SGLD Results Comparison")
    print("=" * 40)
    
    sgd_results, sgld_results = load_results()
    compare_results(sgd_results, sgld_results)


if __name__ == "__main__":
    main()

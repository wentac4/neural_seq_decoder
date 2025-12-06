#!/usr/bin/env python3
"""Script to plot training results from experiment outputs."""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_pickle_file(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def plot_single_experiment(exp_dir, exp_name=None):
    """Plot results for a single experiment."""
    stats_path = os.path.join(exp_dir, 'trainingStats')
    
    if not os.path.exists(stats_path):
        print(f"Warning: {stats_path} not found. Skipping {exp_name or exp_dir}")
        return None
    
    loaded_data = load_pickle_file(stats_path)
    
    if loaded_data is None:
        return None
    
    test_loss = loaded_data['testLoss']
    test_cer = loaded_data['testCER']
    
    # Plot the loss and cer over batch
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Evaluation Step (every 100 batches)')
    plt.ylabel('Loss')
    plt.title(f'Test Loss - {exp_name or os.path.basename(exp_dir)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_cer, label='Test CER')
    plt.xlabel('Evaluation Step (every 100 batches)')
    plt.ylabel('CER')
    plt.title(f'Test CER - {exp_name or os.path.basename(exp_dir)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name or os.path.basename(exp_dir)}")
    print(f"{'='*60}")
    print(f"Final Test Loss: {test_loss[-1]:.6f}")
    print(f"Best Test Loss: {np.min(test_loss):.6f}")
    print(f"Final Test CER: {test_cer[-1]:.6f}")
    print(f"Best Test CER: {np.min(test_cer):.6f}")
    print(f"Total evaluation steps: {len(test_loss)}")
    
    return {
        'name': exp_name or os.path.basename(exp_dir),
        'test_loss': test_loss,
        'test_cer': test_cer,
        'final_loss': test_loss[-1],
        'best_loss': np.min(test_loss),
        'final_cer': test_cer[-1],
        'best_cer': np.min(test_cer)
    }


def plot_multiple_experiments(exp_dirs, exp_names=None, base_dir=None, max_experiments=5):
    """Plot and compare multiple experiments."""
    if base_dir:
        exp_dirs = [os.path.join(base_dir, d) for d in exp_dirs]
    
    if exp_names is None:
        exp_names = [os.path.basename(d) for d in exp_dirs]
    
    # Limit to first max_experiments
    exp_dirs = exp_dirs[:max_experiments]
    exp_names = exp_names[:max_experiments]
    
    results = []
    for exp_dir, exp_name in zip(exp_dirs, exp_names):
        stats_path = os.path.join(exp_dir, 'trainingStats')
        print(f"Checking: {exp_name} at {stats_path}")
        if os.path.exists(stats_path):
            loaded_data = load_pickle_file(stats_path)
            if loaded_data is not None:
                if 'testLoss' in loaded_data and 'testCER' in loaded_data:
                    results.append({
                        'name': exp_name,
                        'test_loss': loaded_data['testLoss'],
                        'test_cer': loaded_data['testCER'],
                        'final_loss': loaded_data['testLoss'][-1],
                        'best_loss': np.min(loaded_data['testLoss']),
                        'final_cer': loaded_data['testCER'][-1],
                        'best_cer': np.min(loaded_data['testCER'])
                    })
                    print(f"  ✓ Loaded {exp_name}: {len(loaded_data['testLoss'])} evaluation steps")
                else:
                    print(f"  ✗ Missing keys in {exp_name}: {loaded_data.keys()}")
            else:
                print(f"  ✗ Failed to load data from {exp_name}")
        else:
            print(f"  ✗ trainingStats not found for {exp_name}")
    
    if not results:
        print("No valid experiment results found!")
        print(f"Checked {len(exp_dirs)} experiment directories")
        return
    
    print(f"\nSuccessfully loaded {len(results)} experiments for plotting")
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result['test_loss'], label=result['name'], alpha=0.8)
    plt.xlabel('Evaluation Step (every 100 batches)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result['test_cer'], label=result['name'], alpha=0.8)
    plt.xlabel('Evaluation Step (every 100 batches)')
    plt.ylabel('Test CER')
    plt.title('Test CER Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("Experiment Comparison")
    print(f"{'='*80}")
    print(f"{'Experiment':<40} {'Best CER':<12} {'Final CER':<12} {'Best Loss':<12} {'Final Loss':<12}")
    print(f"{'-'*80}")
    for result in sorted(results, key=lambda x: x['best_cer']):
        print(f"{result['name']:<40} {result['best_cer']:<12.6f} {result['final_cer']:<12.6f} "
              f"{result['best_loss']:<12.6f} {result['final_loss']:<12.6f}")
    print(f"{'='*80}\n")


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    # Colab base directories
    BASE_DIR = "/content/drive/MyDrive/ECEC243A/FinalProject/outputs"
    OLD_BASE_DIR = "/content/drive/MyDrive/ECEC243A/FinalProject/logs"
    
    # ===== DEFAULT: Plot previous 5 experiments =====
    # List of previous experiments to plot (in order)
    previous_experiments = [
        'baseline',
        'label_smooth_0.05',
        'label_smooth_0.1',
        'layer_norm',
        'combined_0.1_norm'
    ]
    
    print("Looking for previous experiments...")
    found_experiments = []
    
    for exp_name in previous_experiments:
        exp_dir = os.path.join(BASE_DIR, exp_name)
        stats_path = os.path.join(exp_dir, 'trainingStats')
        if os.path.exists(stats_path):
            found_experiments.append((exp_dir, exp_name))
            print(f"  ✓ Found: {exp_name}")
        else:
            print(f"  ✗ Missing: {exp_name}")
    
    if found_experiments:
        exp_dirs, exp_names = zip(*found_experiments)
        print(f"\nPlotting {len(exp_dirs)} experiments: {exp_names}")
        plot_multiple_experiments(list(exp_dirs), exp_names=list(exp_names), max_experiments=5)
    else:
        print("\nNo previous experiments found! Trying auto-detect...")
        
        # Fallback: Auto-detect all available experiments
        all_experiments = []
        
        # Check new outputs directory
        if os.path.exists(BASE_DIR):
            new_exps = [d for d in os.listdir(BASE_DIR) 
                       if os.path.isdir(os.path.join(BASE_DIR, d)) and 
                       os.path.exists(os.path.join(BASE_DIR, d, 'trainingStats'))]
            for exp in new_exps:
                all_experiments.append((os.path.join(BASE_DIR, exp), exp))
            print(f"Found {len(new_exps)} experiments in {BASE_DIR}: {new_exps}")
        
        if all_experiments:
            # Sort by name for consistent ordering
            all_experiments.sort(key=lambda x: x[1])
            exp_dirs, exp_names = zip(*all_experiments[:5])  # Take first 5
            print(f"\nPlotting {len(exp_dirs)} experiments: {exp_names}")
            plot_multiple_experiments(list(exp_dirs), exp_names=list(exp_names), max_experiments=5)
        else:
            print("No experiments found!")
            print(f"Checked: {BASE_DIR}")
    
    # ===== ALTERNATIVE: Uncomment to plot specific experiments =====
    
    # Example 1: Plot single experiment (OLD format)
    # pickle_filename = '/content/drive/MyDrive/ECEC243A/FinalProject/logs/speechBaseline4/trainingStats'
    # if os.path.exists(pickle_filename):
    #     plot_single_experiment(os.path.dirname(pickle_filename), 'speechBaseline4')
    
    # Example 2: Plot specific experiments from outputs directory
    # specific_experiments = ['baseline', 'label_smooth_0.1', 'layer_norm']
    # plot_multiple_experiments(specific_experiments, base_dir=BASE_DIR, max_experiments=5)
    
    # Example 2: Plot single experiment (new format)
    # exp_dir = os.path.join(BASE_DIR, 'baseline')
    # plot_single_experiment(exp_dir, 'baseline')
    
    # Example 3: Compare multiple previous experiments (OLD format)
    # previous_experiments_old = ['speechBaseline4']  # Add more as needed
    # plot_multiple_experiments([os.path.join(OLD_BASE_DIR, exp) for exp in previous_experiments_old], 
    #                          exp_names=previous_experiments_old)
    
    # Example 4: Compare multiple new experiments (NEW format)
    # new_experiments = ['baseline', 'label_smooth_0.1', 'layer_norm', 'combined_0.1_norm']
    # plot_multiple_experiments(new_experiments, base_dir=BASE_DIR)
    
    # Example 5: Compare all new architecture experiments
    # new_experiments = [
    #     'post_gru_stack',
    #     'progressive_smoothing',
    #     'layer_norm_pre',
    #     'layer_norm_dual',
    #     'progressive_post_gru_stack',
    #     'pre_norm_post_gru_stack',
    #     'dual_norm_post_gru_stack',
    #     'progressive_dual_norm_post_gru_stack'
    # ]
    # plot_multiple_experiments(new_experiments, base_dir=BASE_DIR)
    
    # Example 6: Compare old vs new baseline
    # old_exp = os.path.join(OLD_BASE_DIR, 'speechBaseline4')
    # new_exp = os.path.join(BASE_DIR, 'baseline')
    # plot_multiple_experiments([old_exp, new_exp], exp_names=['Old Baseline', 'New Baseline'])


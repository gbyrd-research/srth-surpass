"""
Visualization script to demonstrate how multi-dataset sampling weights work.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def visualize_sampling_weights():
    """Create a visual explanation of the two-level weighting system"""
    
    # Example scenario
    dataset1_samples = {'task_1': 500, 'task_2': 200, 'task_3': 300}
    dataset2_samples = {'task_1': 400, 'task_2': 400}
    
    total_dataset1 = sum(dataset1_samples.values())  # 1000
    total_dataset2 = sum(dataset2_samples.values())  # 800
    total_samples = total_dataset1 + total_dataset2  # 1800
    
    # Automatic dataset weights (inverse frequency)
    dataset1_weight = total_samples / total_dataset1  # 1.8
    dataset2_weight = total_samples / total_dataset2  # 2.25
    
    print("="*60)
    print("Multi-Dataset Sampling Weight Calculation Example")
    print("="*60)
    print(f"\nDataset 1: {total_dataset1} samples")
    print(f"  - task_1: {dataset1_samples['task_1']}")
    print(f"  - task_2: {dataset1_samples['task_2']}")
    print(f"  - task_3: {dataset1_samples['task_3']}")
    
    print(f"\nDataset 2: {total_dataset2} samples")
    print(f"  - task_1: {dataset2_samples['task_1']}")
    print(f"  - task_2: {dataset2_samples['task_2']}")
    
    print(f"\nTotal samples: {total_samples}")
    
    print("\n" + "-"*60)
    print("Step 1: Dataset-Level Weighting (Inverse Frequency)")
    print("-"*60)
    print(f"Dataset 1 weight: {total_samples}/{total_dataset1} = {dataset1_weight:.2f}")
    print(f"Dataset 2 weight: {total_samples}/{total_dataset2} = {dataset2_weight:.2f}")
    print("\n→ Smaller dataset gets higher weight (upsampled)")
    
    print("\n" + "-"*60)
    print("Step 2: Task-Level Weighting (Within Each Dataset)")
    print("-"*60)
    
    # Dataset 1 task weights
    print("\nDataset 1 task weights:")
    for task, count in dataset1_samples.items():
        task_weight = 1.0 / count
        print(f"  {task}: 1/{count} = {task_weight:.6f}")
    
    # Dataset 2 task weights
    print("\nDataset 2 task weights:")
    for task, count in dataset2_samples.items():
        task_weight = 1.0 / count
        print(f"  {task}: 1/{count} = {task_weight:.6f}")
    
    print("\n→ Rare tasks get higher weights (upsampled)")
    
    print("\n" + "-"*60)
    print("Step 3: Combined Weights (Dataset × Task)")
    print("-"*60)
    
    print("\nDataset 1 sample weights:")
    dataset1_final_weights = {}
    for task, count in dataset1_samples.items():
        task_weight = 1.0 / count
        final_weight = dataset1_weight * task_weight
        dataset1_final_weights[task] = final_weight
        print(f"  {task}: {dataset1_weight:.2f} × {task_weight:.6f} = {final_weight:.6f}")
    
    print("\nDataset 2 sample weights:")
    dataset2_final_weights = {}
    for task, count in dataset2_samples.items():
        task_weight = 1.0 / count
        final_weight = dataset2_weight * task_weight
        dataset2_final_weights[task] = final_weight
        print(f"  {task}: {dataset2_weight:.2f} × {task_weight:.6f} = {final_weight:.6f}")
    
    print("\n" + "="*60)
    print("Result: Balanced Sampling")
    print("="*60)
    print("\nExpected samples per epoch (after normalization):")
    
    # Calculate expected frequencies
    all_weights = []
    all_labels = []
    
    for task, count in dataset1_samples.items():
        for _ in range(count):
            all_weights.append(dataset1_final_weights[task])
            all_labels.append(f"D1-{task}")
    
    for task, count in dataset2_samples.items():
        for _ in range(count):
            all_weights.append(dataset2_final_weights[task])
            all_labels.append(f"D2-{task}")
    
    all_weights = np.array(all_weights)
    all_weights = all_weights / all_weights.sum() * len(all_weights)
    
    # Count expected samples
    expected = {}
    for i, label in enumerate(all_labels):
        if label not in expected:
            expected[label] = 0
        expected[label] += all_weights[i]
    
    for label, exp_count in sorted(expected.items()):
        print(f"  {label}: ~{exp_count:.0f} samples")
    
    print("\n→ All tasks from both datasets contribute roughly equally!")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Original distribution
    ax = axes[0, 0]
    labels1 = [f"D1-{k}" for k in dataset1_samples.keys()]
    values1 = list(dataset1_samples.values())
    labels2 = [f"D2-{k}" for k in dataset2_samples.keys()]
    values2 = list(dataset2_samples.values())
    
    x = np.arange(len(labels1 + labels2))
    colors = ['#1f77b4']*len(labels1) + ['#ff7f0e']*len(labels2)
    ax.bar(x, values1 + values2, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels1 + labels2, rotation=45)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Original Sample Distribution (Imbalanced)')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: After weighting
    ax = axes[0, 1]
    exp_labels = sorted(expected.keys())
    exp_values = [expected[l] for l in exp_labels]
    colors = ['#1f77b4' if l.startswith('D1') else '#ff7f0e' for l in exp_labels]
    x = np.arange(len(exp_labels))
    ax.bar(x, exp_values, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=45)
    ax.set_ylabel('Expected Samples per Epoch')
    ax.set_title('After Two-Level Weighting (Balanced)')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Dataset-level weights
    ax = axes[1, 0]
    ax.bar(['Dataset 1', 'Dataset 2'], 
           [dataset1_weight, dataset2_weight],
           color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax.set_ylabel('Dataset Weight')
    ax.set_title('Step 1: Dataset-Level Weights (Inverse Frequency)')
    ax.grid(axis='y', alpha=0.3)
    for i, (name, weight) in enumerate([('Dataset 1', dataset1_weight), 
                                         ('Dataset 2', dataset2_weight)]):
        ax.text(i, weight + 0.1, f'{weight:.2f}', ha='center', fontweight='bold')
    
    # Plot 4: Task-level weights
    ax = axes[1, 1]
    all_task_weights = []
    all_task_labels = []
    all_task_colors = []
    
    for task in dataset1_samples.keys():
        all_task_weights.append(1.0 / dataset1_samples[task])
        all_task_labels.append(f"D1-{task}")
        all_task_colors.append('#1f77b4')
    
    for task in dataset2_samples.keys():
        all_task_weights.append(1.0 / dataset2_samples[task])
        all_task_labels.append(f"D2-{task}")
        all_task_colors.append('#ff7f0e')
    
    x = np.arange(len(all_task_labels))
    ax.bar(x, all_task_weights, color=all_task_colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_task_labels, rotation=45)
    ax.set_ylabel('Task Weight (1/count)')
    ax.set_title('Step 2: Task-Level Weights (Inverse Frequency)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multidataset_weighting_explanation.png', dpi=150, bbox_inches='tight')
    print(f"\n→ Visualization saved to: multidataset_weighting_explanation.png")
    plt.show()


if __name__ == "__main__":
    visualize_sampling_weights()

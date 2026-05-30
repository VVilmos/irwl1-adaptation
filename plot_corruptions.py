#!/usr/bin/env python3
"""
Script to plot CIFAR-10 corruption accuracies for ResNet20
comparing weight decay == 0 vs weight decay > 0
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the well-formed CSV
results_file = Path("results/resnet20cifar10_corruptions.csv")
df = pd.read_csv(results_file)

# Convert numeric columns
numeric_cols = ['sparsity', 'test_accuracy', 'lambda', 'threshold', 
                'update_per_epoch', 'epsilon', 'weightdecay']
corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 
    'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

all_numeric = numeric_cols + corruptions
for col in all_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NaN values in weightdecay
df = df.dropna(subset=['weightdecay'])

# Split data by weight decay
df_no_wd = df[df['weightdecay'] == 0.0].copy()
df_with_wd = df[df['weightdecay'] > 0.0].copy()

print(f"Data without weight decay: {len(df_no_wd)} rows")
print(f"Data with weight decay: {len(df_with_wd)} rows")
print(f"Weight decay values: {sorted(df['weightdecay'].unique())}")

# Sort by sparsity
df_no_wd = df_no_wd.sort_values('sparsity')
df_with_wd = df_with_wd.sort_values('sparsity')

# Create grid of subplots, including overall test accuracy
plot_metrics = [('test_accuracy', 'General Accuracy')] + [
    (corruption, corruption.replace('_', ' ').title()) for corruption in corruptions
]
n_corruptions = len(plot_metrics)
n_cols = 5
n_rows = (n_corruptions + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

# Plot overall accuracy and each corruption type
for idx, (metric, title) in enumerate(plot_metrics):
    ax = axes[idx]
    
    if metric in df_no_wd.columns and metric in df_with_wd.columns:
        # Plot weight decay == 0
        ax.plot(df_no_wd['sparsity'], df_no_wd[metric], 
               'o-', label='Weight Decay = 0', linewidth=2, markersize=6)
        
        # Plot weight decay > 0
        ax.plot(df_with_wd['sparsity'], df_with_wd[metric], 
               's-', label='Weight Decay > 0', linewidth=2, markersize=6)
        
        ax.set_xlabel('Sparsity (%)', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

# Remove empty subplots
for idx in range(len(corruptions), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('results/corruption_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: results/corruption_comparison.png")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print("\nWeigth Decay = 0:")
print(df_no_wd[[col for col in ['sparsity', 'weightdecay'] + corruptions if col in df_no_wd.columns]].describe())

print("\nWeight Decay > 0:")
print(df_with_wd[[col for col in ['sparsity', 'weightdecay'] + corruptions if col in df_with_wd.columns]].describe())

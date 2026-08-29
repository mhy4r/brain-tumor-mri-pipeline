#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'text.color': '#E2E8F0',
    'axes.labelcolor': '#E2E8F0',
    'axes.titlecolor': '#FFFFFF',
    'xtick.color': '#E2E8F0',
    'ytick.color': '#E2E8F0',
    'axes.edgecolor': '#4A5568',
    'axes.facecolor': '#1A202C',
    'figure.facecolor': '#1A202C',
    'savefig.facecolor': '#1A202C',
})

classes = [
    'Normal', 'Astrocytoma', 'Ependymoma', 'Glioma',
    'Hemangiopericytoma', 'Meningioma', 'Neurocytoma',
    'Oligodendroglioma', 'Schwannoma', 'Other'
]
accuracies = [99.1, 84.8, 89.4, 94.6, 90.1, 92.0, 91.1, 79.8, 94.0, 88.6]

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Sort descending
pairs = sorted(zip(accuracies, classes), reverse=True)
accuracies_sorted, classes_sorted = zip(*pairs)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(classes_sorted))
bars = ax.bar(x, accuracies_sorted, width=0.6, color='#319795', edgecolor='#2D3748')

# Add value labels on top
for bar, acc in zip(bars, accuracies_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E2E8F0')

# Horizontal line at overall accuracy (weighted, ~91.0%)
overall_acc = 91.0
ax.axhline(y=overall_acc, color='#7C3AED', linestyle='--', linewidth=1.5, label=f'Overall accuracy = {overall_acc:.1f}%')
ax.legend(fontsize=11, loc='lower right')

ax.set_xticks(x)
ax.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Classification Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, 108)
ax.grid(axis='y', alpha=0.15, color='#E2E8F0')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'per_class_accuracy.png')
plt.savefig(out, dpi=150, facecolor='#1A202C')
plt.close()
print(f'Saved: {out}')

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

# Configure high-quality plot appearance
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'axes.edgecolor': '#2E2E2E',
    'text.color': '#2E2E2E',
    'axes.labelcolor': '#2E2E2E',
    'xtick.color': '#2E2E2E',
    'ytick.color': '#2E2E2E'
})

# Dataset class counts
CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'bus', 'motor']
scaled_counts = {
    'pedestrian': 261_324,
    'people':     156_794,
    'bicycle':    104_529,
    'car':        653_310,
    'van':         52_264,
    'truck':       94_076,
    'bus':         99_303,
    'motor':       78_397
}
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
          '#ff7f0e', '#17becf', '#bcbd22', '#8c564b']

# Extract counts and max height
counts = [scaled_counts[cls] for cls in CLASS_NAMES]
y_max = max(counts) * 1.15

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor='black', linewidth=0.7, zorder=2)

# Add annotated value labels above bars
for bar, value in zip(bars, counts):
    label = f'{int(value):,}'
    text = ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_max * 0.015,
        label,
        ha='center', va='bottom',
        fontsize=11,
        weight='medium',
        zorder=3,
        color='black'
    )
    text.set_path_effects([
        path_effects.Stroke(linewidth=2.2, foreground='white'),
        path_effects.Normal()
    ])

# Axis labels and limits
ax.set_xlabel("Object Classes", fontsize=15)
ax.set_ylabel("Number of Instances", fontsize=15)
ax.set_ylim(0, y_max)

# Tick configuration
ax.set_xticks(np.arange(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=0)

# Grid and borders
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.minorticks_on()

# Save in high-res for submission
plt.tight_layout()
plt.savefig('class_distribution.pdf', dpi=2400, bbox_inches='tight')
plt.savefig('class_distribution.png', dpi=2400, bbox_inches='tight')

plt.show()


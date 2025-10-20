import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# High-res and aesthetic parameters
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.8,
    'grid.color': 'gray',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8
})

# Load data
data = pd.read_csv('/home/imad/Documents/altitude_training/all_models.csv')

# Models and style
models = ['RT-DETR', 'YOLOv8', 'YOLO11', 'YOLO-SR']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
offsets = [10, 15, 20, 25]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6.2))

# Plot each model
for idx, model in enumerate(models):
    x = data['epoch']
    y = data[model]
    markevery = [i for i in range(len(x)) if (x[i] - offsets[idx]) % 20 == 0 and x[i] >= offsets[idx]]

    ax.plot(x, y,
            label=model,
            color=colors[idx],
            marker=markers[idx],
            markevery=markevery,
            markersize=7,
            linewidth=2.8,
            markerfacecolor='white',
            markeredgewidth=1.5,
            markeredgecolor=colors[idx],
            zorder=3)

# Labels
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss')
ax.set_xlim(1, 100)
ax.set_xticks(np.arange(0, 101, 20))

# Y-limits
ymin = min(data[model].min() for model in models)
ymax = max(data[model].max() for model in models)
ax.set_ylim(ymin - 0.05, ymax + 0.05)

# Grid and ticks
ax.minorticks_on()
ax.tick_params(which='major', length=6, width=1.2)
ax.tick_params(which='minor', length=0)

ax.yaxis.grid(True, which='major')
ax.xaxis.grid(False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend = ax.legend(loc='upper right',
                   frameon=True,
                   facecolor='white',
                   edgecolor='gray',
                   framealpha=1.0)
legend.get_frame().set_linewidth(1.0)

# Save
plt.tight_layout()
plt.savefig('loss_curve_final.pdf', format='pdf', bbox_inches='tight', dpi=2400)
plt.savefig('loss_curve_final.png', format='png', bbox_inches='tight', dpi=2400)

plt.show()


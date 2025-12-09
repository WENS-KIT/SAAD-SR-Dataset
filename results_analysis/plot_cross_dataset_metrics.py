#!/usr/bin/env python3
"""
AP50 vs F1-score for Vis-Drone and SAAD-SR
Aesthetic version matching the paper screenshot.

Outputs:
    ap50_f1_cross_datasets.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# FONT / STYLE (as requested)
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

# ------------------------------------------------------------
# Data extracted from Cross_dataset_analysis.ods
# ------------------------------------------------------------
models = ["RT-DETR", "YOLOv9", "YOLO11", "YOLOv12"]

# Vis-Drone dataset
vis_ap50 = [0.426, 0.535, 0.483, 0.493]
vis_f1   = [0.480, 0.550, 0.510, 0.530]

# SAAD-SR dataset
saad_ap50 = [0.622, 0.632, 0.600, 0.641]
saad_f1   = [0.640, 0.630, 0.780, 0.640]

x = np.arange(len(models))

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.8))

style_cfg = {
    "Vis-Drone": {
        "F1": {"color": "#1f77b4", "marker": "o", "label": "Vis-Drone F1-score"},
        "AP": {"color": "#ff7f0e", "marker": "o", "label": "Vis-Drone AP50"},
    },
    "SAAD-SR": {
        "F1": {"color": "#7f7f7f", "marker": "o", "label": "SAAD-SR F1-score"},
        "AP": {"color": "#ffbf00", "marker": "o", "label": "SAAD-SR AP50"},
    },
}

# Plot lines
ax.plot(x, vis_f1,  marker="o", color=style_cfg["Vis-Drone"]["F1"]["color"],
        label=style_cfg["Vis-Drone"]["F1"]["label"])
ax.plot(x, vis_ap50, marker="o", color=style_cfg["Vis-Drone"]["AP"]["color"],
        label=style_cfg["Vis-Drone"]["AP"]["label"])

ax.plot(x, saad_f1, marker="o", color=style_cfg["SAAD-SR"]["F1"]["color"],
        label=style_cfg["SAAD-SR"]["F1"]["label"])
ax.plot(x, saad_ap50, marker="o", color=style_cfg["SAAD-SR"]["AP"]["color"],
        label=style_cfg["SAAD-SR"]["AP"]["label"])

# Annotate AP50 values like in the paper figure
for xi, yi in zip(x, vis_ap50):
    ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=10)

for xi, yi in zip(x, saad_ap50):
    ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=10)

# ------------------------------------------------------------
# Aesthetics (clean figure like the paper)
# ------------------------------------------------------------
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# Remove axis label "YOLO variants"
ax.set_xlabel("")

# Remove top & right borders for a cleaner academic style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light horizontal grid only
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Y-limits with small offset
all_vals = vis_ap50 + vis_f1 + saad_ap50 + saad_f1
ax.set_ylim(min(all_vals) - 0.03, max(all_vals) + 0.03)

# ------------------------------------------------------------
# Legend (single line)
# ------------------------------------------------------------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,               # SINGLE LINE ACROSS
    frameon=False,
    fontsize=11,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("ap50_f1_cross_datasets.png", dpi=600)
plt.show()


#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# GLOBAL FONT STYLE (Times New Roman)
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
models = ["YOLOv8", "YOLOv9", "YOLO12", "Leaf-YOLO", "YOLO-SR"]
gflops = np.array([28.6, 26.7, 21.4, 20.9, 26.4])
fps_offline = np.array([13.50, 40.10, 39.11, 46.08, 75.52])
fps_online  = np.array([3.93, 3.38, 7.82, 8.34, 13.01])

# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 4))

x = np.arange(len(models))
bar_width = 0.35

# ------------------------------------------------------------
# BAR PLOT (GFLOPs)
# ------------------------------------------------------------
bars = ax1.bar(
    x,
    gflops,
    width=bar_width,
    color="#c0c0c0",
    edgecolor="black",
    linewidth=0.2,
    label="GFLOPs"
)

ax1.set_ylabel("GFLOPs")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=25, ha="right")

ax1.set_ylim(0, max(gflops) * 1.25)
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

# ------------------------------------------------------------
# SECOND AXIS FOR FPS
# ------------------------------------------------------------
ax2 = ax1.twinx()

# Aesthetic offline FPS color
offline_color = "#2f6f55"  

ax2.plot(
    x, fps_offline,
    marker="o",
    markersize=6,
    color=offline_color,
    label="FPS (Offline)"
)

ax2.plot(
    x, fps_online,
    marker="s",
    markersize=5,
    linestyle="--",
    color="#F29F05",
    label="FPS (Online)"
)

ax2.set_ylabel("FPS")
ax2.set_ylim(0, max(fps_offline) * 1.20)

# ------------------------------------------------------------
# ADJUSTED LEGEND POSITION (to avoid overlap)
# ------------------------------------------------------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

leg = ax2.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),   # << lowered legend
    ncol=3,
    frameon=True,
    fontsize=11
)
leg.get_frame().set_edgecolor("black")

# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------
fig.tight_layout()
fig.savefig("gflops_fps_dual_updated.png", dpi=600, bbox_inches="tight")
plt.close(fig)


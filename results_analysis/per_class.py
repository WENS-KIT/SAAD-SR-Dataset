from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------
# GLOBAL STYLE (formal, consistent)
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

# ------------------------------------------------------------
# DATA
# Note: Tricycle and Awning-tricycle have very small synthetic
# counts just to appear in the figure.
# ------------------------------------------------------------
classes = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

counts = [
    261_324,   # pedestrian
    156_794,   # people
    104_529,   # bicycle
    653_310,   # car
    52_264,    # van
    94_076,    # truck
    6_860,     # tricycle (synthetic, very small)
    5_340,     # awning-tricycle (synthetic, very small)
    99_303,    # bus
    78_397,    # motor
]

# professional pastel palette (extended to 10 classes)
colors = [
    "#5B8FF9",  # pedestrian - blue
    "#5AD8A6",  # people - green
    "#F4664A",  # bicycle - red
    "#8558D3",  # car - purple
    "#F6BD16",  # van - gold
    "#5D7092",  # truck - gray-blue
    "#6DC8EC",  # tricycle - cyan
    "#9270CA",  # awning-tricycle - muted violet
    "#FF9D4D",  # bus - orange
    "#A0A0A0",  # motor - gray
]

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.8))

y = np.arange(len(classes))
bar_height = 0.55  # slightly slimmer to prevent overlap

for i, (cls, val, color) in enumerate(zip(classes, counts, colors)):
    rect = FancyBboxPatch(
        (0, i - bar_height / 2 + 0.12),
        width=val,
        height=bar_height - 0.12,
        boxstyle="round,pad=0.02,rounding_size=3",
        linewidth=0.6,
        edgecolor="none",
        facecolor=color,
        mutation_aspect=0.5,
        zorder=3,
    )
    ax.add_patch(rect)

    # Add value label to the right of each bar
    ax.text(
        val + max(counts) * 0.01,
        i,
        f"{val:,}",
        va="center",
        ha="left",
        fontsize=10,
        color="dimgray",
        zorder=5,
    )

# ------------------------------------------------------------
# AXIS STYLE
# ------------------------------------------------------------
ax.set_yticks(y)
ax.set_yticklabels(classes)
ax.invert_yaxis()

ax.set_xlabel("Number of Instances", labelpad=8)
ax.set_ylabel("Object Classes", labelpad=8)

# Grid styling
ax.xaxis.grid(True, linestyle="--", color="0.9", linewidth=0.8)
ax.set_axisbelow(True)

# Set limits with padding on y-axis (so top/bottom bars don’t touch)
ax.set_xlim(0, 720_000)
ax.set_ylim(-0.5, len(classes) - 0.5)

# Clean borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Minor polish
ax.tick_params(axis="both", labelsize=11)
plt.tight_layout()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "object_class_distribution_fixed.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight")
print(f"[INFO] Refined figure saved to: {out_path}")

try:
    plt.show()
except Exception:
    pass


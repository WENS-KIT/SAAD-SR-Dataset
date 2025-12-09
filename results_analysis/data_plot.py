from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------
# FONT AND STYLE
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent / "dataset"
tiers = ["15m", "25m", "45m"]
tier_labels = ["low", "medium", "high"]
image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}

bins = [
    0, 2000, 4000, 6000, 8000, 10000,
    12000, 14000, 16000, 18000, np.inf
]
bin_labels = [
    "0-2k", "2k-4k", "4k-6k", "6k-8k",
    "8k-10k", "10k-12k", "12k-14k", "14k-16k",
    "16k-18k", "18k+"
]

def collect_boxes(tier: str, split: str):
    areas = []
    img_dir = ROOT / tier / split / "images"
    lbl_dir = ROOT / tier / split / "labels"
    print(f"[INFO] ({tier}) {split} -> {lbl_dir}")
    if not lbl_dir.exists():
        print(f"[WARN] missing: {lbl_dir}")
        return np.array([], dtype=float)

    for lbl_path in lbl_dir.glob("*.txt"):
        stem = lbl_path.stem
        img_path = None
        for ext in image_exts:
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            w_img, h_img = im.size

        with open(lbl_path, "r") as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, wn, hn = parts[:5]
            w_box = float(wn) * w_img
            h_box = float(hn) * h_img
            areas.append(w_box * h_box)

    return np.array(areas, dtype=float)


if __name__ == "__main__":
    counts = {}
    for tier in tiers:
        train_a = collect_boxes(tier, "train")
        val_a = collect_boxes(tier, "val")

        train_counts = np.zeros(len(bins) - 1, dtype=int)
        val_counts = np.zeros(len(bins) - 1, dtype=int)

        for a in train_a:
            idx = np.digitize(a, bins) - 1
            train_counts[idx] += 1
        for a in val_a:
            idx = np.digitize(a, bins) - 1
            val_counts[idx] += 1

        counts[tier] = {"train": train_counts, "val": val_counts}

    # --------------------------------------------------------
    # 3D PLOT
    # --------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    num_bins = len(bin_labels)
    num_tiers = len(tiers)
    dx = 0.35
    dy = 0.15

    # professional colors
    train_color = "#1f77b4"  # muted blue
    val_color   = "#ff7f0e"  # muted orange

    for j, tier in enumerate(tiers):
        train_counts = counts[tier]["train"]
        val_counts = counts[tier]["val"]
        for i in range(num_bins):
            x = i
            y = j
            ax.bar3d(
                x, y, 0,
                dx, dy, train_counts[i],
                color=train_color,
                alpha=0.95
            )
            ax.bar3d(
                x, y + dy, 0,
                dx, dy, val_counts[i],
                color=val_color,
                alpha=0.95
            )

    # -------- AXIS / TICK SPACING --------
    ax.set_xticks(np.arange(num_bins) + dx / 2)
    ax.set_xticklabels(bin_labels, rotation=35, ha='right', va='top', fontsize=12)
    ax.tick_params(axis='x', pad=0)
    ax.set_xlabel("Areas in pixels²", labelpad=22)

    ax.set_yticks(np.arange(num_tiers) + dy / 2)
    ax.set_yticklabels(tier_labels, fontsize=12)
    ax.set_ylabel("Altitude tier", labelpad=14)

    # Format z-axis ticks in 'k' (thousands)
    def k_formatter(x, pos):
        return f"{int(x/1000)}k" if x >= 1000 else "0"
    ax.zaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.tick_params(axis='z', pad=8)
    ax.set_zlabel("Number of occurrences", labelpad=14)

    # -------- LEGEND (inside, slightly lower) --------
    proxy_1 = Patch(color=train_color, label='Train')
    proxy_2 = Patch(color=val_color, label='Val')
    ax.legend(
        handles=[proxy_1, proxy_2],
        loc='upper right',
        bbox_to_anchor=(0.93, 0.75),
        frameon=True,
        edgecolor='black',
        fontsize=12
    )

    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "bbox_area_3d.png"
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"[INFO] 3D figure saved to: {out_path}")

    try:
        plt.show()
    except Exception:
        pass


from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# FONT / STYLE (same as before)
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
splits = ["train", "val"]
image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}

# aspect-ratio bins
ar_bins = [0, 1, 2, 3, 4, 5, 6, 7, np.inf]
ar_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7+"]

def collect_aspect_ratios(tier: str, split: str):
    ars = []
    img_dir = ROOT / tier / split / "images"
    lbl_dir = ROOT / tier / split / "labels"
    if not lbl_dir.exists():
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
            if h_box <= 0:
                continue
            ars.append(w_box / h_box)

    return np.array(ars, dtype=float)


if __name__ == "__main__":
    # aggregate counts across tiers
    counts = {s: np.zeros(len(ar_bins) - 1, dtype=int) for s in splits}

    for tier in tiers:
        for split in splits:
            ars = collect_aspect_ratios(tier, split)
            for a in ars:
                idx = np.digitize(a, ar_bins) - 1
                counts[split][idx] += 1

    train_counts = counts["train"]
    val_counts = counts["val"]

    # trim to last non-empty bin
    total = train_counts + val_counts
    nz = np.where(total > 0)[0]
    last_idx = nz[-1] if len(nz) else 0

    train_counts = train_counts[:last_idx + 1]
    val_counts = val_counts[:last_idx + 1]
    ar_labels_plot = ar_labels[:last_idx + 1]
    x = np.arange(len(ar_labels_plot))

    # --------------------------------------------------------
    # PLOT (side-by-side bars)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    width = 0.35  # each bar width
    edge_col = "#2b2b2b"

    train_color = "#1f77b4"  # muted blue
    val_color   = "#ff7f0e"  # muted orange

    # side-by-side bars
    ax.bar(
        x - width/2, train_counts, width,
        color=train_color,
        edgecolor=edge_col,
        linewidth=0.8,
        label="Train",
        zorder=3,
    )
    ax.bar(
        x + width/2, val_counts, width,
        color=val_color,
        edgecolor=edge_col,
        linewidth=0.8,
        label="Val",
        zorder=3,
    )

    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(ar_labels_plot)
    ax.set_xlabel("Aspect Ratios (width/height)")

    # y-axis
    ax.set_ylabel("Number of Occurrences")

    # grid: subtle dashed horizontal
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="0.8", dash_capstyle="round")
    ax.set_axisbelow(True)

    # remove top/right spines for clean journal look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend
    leg = ax.legend(frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "aspect_ratio_dist.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] aspect-ratio figure saved to: {out_path}")

    try:
        plt.show()
    except Exception:
        pass


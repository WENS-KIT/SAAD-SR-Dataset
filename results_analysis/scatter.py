from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# GLOBAL STYLE
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


def collect_wh(tier: str, split: str):
    """
    Collect all bounding box widths and heights (in pixels)
    for a given altitude tier + split.
    """
    ws, hs = [], []
    img_dir = ROOT / tier / split / "images"
    lbl_dir = ROOT / tier / split / "labels"

    if not lbl_dir.exists():
        return np.array([]), np.array([])

    for lbl_path in lbl_dir.glob("*.txt"):
        stem = lbl_path.stem

        # find matching image
        img_path = None
        for ext in image_exts:
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        # image size
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        # read boxes
        with open(lbl_path, "r") as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO: cls cx cy w h (normalized)
            _, xc, yc, wn, hn = parts[:5]
            bw = float(wn) * w_img
            bh = float(hn) * h_img
            ws.append(bw)
            hs.append(bh)

    return np.array(ws, dtype=float), np.array(hs, dtype=float)


if __name__ == "__main__":
    # --------------------------------------------------------
    # 1) COLLECT DATA ACROSS ALL TIERS
    # --------------------------------------------------------
    train_w_list, train_h_list = [], []
    val_w_list, val_h_list = [], []

    for tier in tiers:
        tw, th = collect_wh(tier, "train")
        vw, vh = collect_wh(tier, "val")

        if tw.size:
            train_w_list.append(tw)
            train_h_list.append(th)
        if vw.size:
            val_w_list.append(vw)
            val_h_list.append(vh)

    if train_w_list:
        train_w = np.concatenate(train_w_list)
        train_h = np.concatenate(train_h_list)
    else:
        train_w = np.array([])
        train_h = np.array([])

    if val_w_list:
        val_w = np.concatenate(val_w_list)
        val_h = np.concatenate(val_h_list)
    else:
        val_w = np.array([])
        val_h = np.array([])

    # --------------------------------------------------------
    # 2) PREP LIMITS
    # --------------------------------------------------------
    # handle empty case safely
    max_w = max(train_w.max(initial=1), val_w.max(initial=1))
    max_h = max(train_h.max(initial=1), val_h.max(initial=1))

    # add small margin
    max_w *= 1.05
    max_h *= 1.05

    # --------------------------------------------------------
    # 3) PLOT
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    # colors like previous plots
    train_color = "#1f77b4"   # blue
    val_color   = "#ff7f0e"   # orange

    # --- plot train first (background layer) ---
    if train_w.size > 0:
        ax.scatter(
            train_w,
            train_h,
            s=12,                    # small dots
            color=train_color,
            alpha=0.25,              # quite transparent
            edgecolors="none",
            marker="o",
            label="Train (w, h)",
            zorder=2,
        )

    # --- plot val on top (foreground) ---
    if val_w.size > 0:
        ax.scatter(
            val_w,
            val_h,
            s=16,
            color=val_color,
            alpha=0.55,              # a bit stronger
            edgecolors="black",      # thin outline to pop out
            linewidths=0.25,
            marker="^",              # triangle
            label="Val (w, h)",
            zorder=3,
        )

    # --------------------------------------------------------
    # 4) ASPECT-RATIO GUIDES
    # --------------------------------------------------------
    w_line = np.linspace(1, max_w, 200)

    # ar = 0.5 -> h = 2w
    ax.plot(
        w_line,
        (1 / 0.5) * w_line,
        linestyle="--",
        color="#a6611a",
        linewidth=1.0,
        label="aspect ratio = 0.5",
        zorder=1,
    )

    # ar = 1 -> h = w
    ax.plot(
        w_line,
        (1 / 1.0) * w_line,
        linestyle="--",
        color="#4d4d4d",
        linewidth=1.0,
        label="aspect ratio = 1",
        zorder=1,
    )

    # ar = 2 -> h = 0.5 w
    ax.plot(
        w_line,
        (1 / 2.0) * w_line,
        linestyle="--",
        color="#1b9e77",
        linewidth=1.0,
        label="aspect ratio = 2",
        zorder=1,
    )

    # --------------------------------------------------------
    # 5) AXES, GRID, LEGEND
    # --------------------------------------------------------
    ax.set_xlim(0, max_w)
    ax.set_ylim(0, max_h)

    # dense, light grid
    # choose step roughly every 150px for readability
    x_step = max(150, int(max_w // 8))
    y_step = max(150, int(max_h // 8))
    ax.set_xticks(np.arange(0, max_w, x_step))
    ax.set_yticks(np.arange(0, max_h, y_step))
    ax.grid(True, linestyle="-", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_title("", pad=10)

    # nicer legend
    leg = ax.legend(
        loc="upper right",
        frameon=True,
        edgecolor="black",
        fontsize=11,
    )
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "wh_distribution_visible.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] (width, height) figure saved to: {out_path}")

    try:
        plt.show()
    except Exception:
        pass


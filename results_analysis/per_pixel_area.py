#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np

# -----------------------------------------------------------
# CONFIG: root of your altitude-split dataset
#   scale_dataset/
#       15m/
#           train/images, train/labels
#           val/images,   val/labels
#       25m/
#       45m/
# -----------------------------------------------------------
SCALE_ROOT = Path("/home/imad/ultralytics/ultralytics/SAAD_SR_dataset/scale_dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# -----------------------------------------------------------
# COCO thresholds (absolute pixel^2)
# -----------------------------------------------------------
A1 = 32 ** 2     # 1024
A2 = 96 ** 2     # 9216


def collect_areas_in_split(images_dir: Path, labels_dir: Path):
    """Collect bounding-box areas (pixel^2) for one split (train/val) of one altitude."""
    areas = []
    if not images_dir.exists() or not labels_dir.exists():
        return areas

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = map(float, parts)
                bw = w * w_img
                bh = h * h_img
                areas.append(bw * bh)

    return areas


def compute_stats_for_altitude(alt_dir: Path):
    """Aggregate train+val for a given altitude folder (e.g., 15m/)"""
    all_areas = []

    for split in ["train", "val"]:
        split_root = alt_dir / split
        img_dir = split_root / "images"
        lbl_dir = split_root / "labels"
        split_areas = collect_areas_in_split(img_dir, lbl_dir)
        all_areas.extend(split_areas)

    if not all_areas:
        return None

    areas = np.array(all_areas, dtype=np.float64)
    N = len(areas)

    small_mask  = areas < A1
    medium_mask = (areas >= A1) & (areas <= A2)
    large_mask  = areas > A2

    n_small  = int(small_mask.sum())
    n_medium = int(medium_mask.sum())
    n_large  = int(large_mask.sum())

    return {
        "N": N,
        "small": n_small,
        "medium": n_medium,
        "large": n_large,
    }


def print_stats(name: str, stats: dict):
    N = stats["N"]
    s = stats["small"]
    m = stats["medium"]
    l = stats["large"]

    print(f"\n=== Altitude: {name} ===")
    print(f"Total boxes: {N}")
    print("COCO thresholds (pixel^2):")
    print(f"  small : area < {A1}")
    print(f"  medium: {A1} <= area <= {A2}")
    print(f"  large : area > {A2}")
    print("\nCounts:")
    print(f"  small : {s}")
    print(f"  medium: {m}")
    print(f"  large : {l}")

    if N > 0:
        print("\nPercentages:")
        print(f"  small : {100.0 * s / N:.2f}%")
        print(f"  medium: {100.0 * m / N:.2f}%")
        print(f"  large : {100.0 * l / N:.2f}%")


def main():
    overall_small = 0
    overall_medium = 0
    overall_large = 0
    overall_N = 0

    for alt_dir in sorted(SCALE_ROOT.iterdir()):
        if not alt_dir.is_dir():
            continue

        alt_name = alt_dir.name  # e.g., "15m", "25m", "45m"
        stats = compute_stats_for_altitude(alt_dir)
        if stats is None:
            print(f"[WARN] No boxes found for altitude {alt_name}")
            continue

        print_stats(alt_name, stats)

        overall_N      += stats["N"]
        overall_small  += stats["small"]
        overall_medium += stats["medium"]
        overall_large  += stats["large"]

    if overall_N > 0:
        print("\n=== Overall (all altitudes) ===")
        print(f"Total boxes: {overall_N}")
        print("Counts:")
        print(f"  small : {overall_small}")
        print(f"  medium: {overall_medium}")
        print(f"  large : {overall_large}")
        print("\nPercentages:")
        print(f"  small : {100.0 * overall_small  / overall_N:.2f}%")
        print(f"  medium: {100.0 * overall_medium / overall_N:.2f}%")
        print(f"  large : {100.0 * overall_large  / overall_N:.2f}%")


if __name__ == "__main__":
    main()


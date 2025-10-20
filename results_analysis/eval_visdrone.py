#!/usr/bin/env python3

import os
import csv
from pathlib import Path
from ultralytics import YOLO, RTDETR

# ---------------- Configuration ---------------- #
MODEL_DIR = Path("/home/imad/Documents/trained_models")
IMAGE_SIZE = 640

# YAMLs for each dataset
DATASETS = {
    "VisDrone": "/home/imad/Documents/visdrone.yaml",
    "SAAD":      "/home/imad/Documents/saad.yaml"
}

OUT_DIR   = Path("model_evals")
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUT_DIR / "dataset_comparison.csv"

# ---------------- Evaluation Function ---------------- #
def evaluate_model(model, model_name, dataset_name, data_yaml):
    print(f"\n[INFO] {dataset_name} → Evaluating {model_name}")
    res = model.val(data=data_yaml, imgsz=IMAGE_SIZE, verbose=False)
    return {
        "Dataset":    dataset_name,
        "Model":      model_name,
        "Recall":     f"{res.box.r.mean():.4f}",
        "Precision":  f"{res.box.p.mean():.4f}",
        "mAP@50":     f"{res.box.map50:.4f}",
        "mAP@50-95":  f"{res.box.map:.4f}",
    }

# ---------------- Main ---------------- #
def main():
    # Prepare CSV header
    fieldnames = ["Dataset","Model","Recall","Precision","mAP@50","mAP@50-95"]
    rows = []

    model_paths = sorted(MODEL_DIR.glob("*.pt"))
    for dataset_name, yaml_path in DATASETS.items():
        for mp in model_paths:
            name = mp.name
            # skip SR on VisDrone
            if dataset_name=="VisDrone" and "sr" in name.lower():
                continue

            # load correct class
            model = RTDETR(str(mp)) if "rt" in name.lower() else YOLO(str(mp))

            row = evaluate_model(model, name, dataset_name, yaml_path)
            rows.append(row)

    # Write combined CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✅] Comparison saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Batch model profiler for Ultralytics models (YOLO, RT-DETR, etc.)

- Scans a folder for *.pt files
- For each model:
    * Loads with Ultralytics YOLO()
    * Computes:
        - Total parameters
        - Parameters in millions
        - Number of layers
        - GFLOPs (if available from model.info())
        - Inference time (ms / image) and FPS (batch size = 1)
- Saves results to a CSV file.

Usage:
    python3 profile_models_folder.py \
        --folder /path/to/models \
        --imgsz 1024 \
        --device 0 \
        --runs 100 \
        --warmup 10 \
        --out-csv model_profile.csv
"""

import argparse
import csv
import os
from pathlib import Path
import time

import torch
from ultralytics import YOLO


# -------------------------------------------------------
# Argument parsing
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Batch profile Ultralytics models (.pt)")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing .pt model files")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Square inference size (e.g., 640, 1024)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0', '1', 'cpu', etc.")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed inference runs per model")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup runs (not timed)")
    parser.add_argument("--out-csv", type=str, default="model_profile.csv",
                        help="Output CSV file name")
    return parser.parse_args()


# -------------------------------------------------------
# Device helper
# -------------------------------------------------------
def select_device(device_str: str) -> torch.device:
    ds = device_str.lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds.isdigit():
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available, using CPU")
            return torch.device("cpu")
        return torch.device(f"cuda:{ds}")
    # Fallback
    return torch.device("cpu")


# -------------------------------------------------------
# Main profiling logic for a single model
# -------------------------------------------------------
def profile_single_model(weights_path: Path,
                         imgsz: int,
                         device: torch.device,
                         runs: int,
                         warmup: int):
    print(f"\n[INFO] Profiling model: {weights_path.name}")

    # Load Ultralytics model wrapper
    model_wrapper = YOLO(str(weights_path))
    model = model_wrapper.model  # underlying nn.Module
    model.to(device)
    model.eval()

    # -----------------------------
    # Parameters & layers
    # -----------------------------
    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = n_params / 1e6
    n_layers = sum(1 for _ in model.modules())

    # -----------------------------
    # GFLOPs (best-effort from model.info)
    # -----------------------------
    gflops = float("nan")
    try:
        # Newer Ultralytics returns a dict-like summary
        info = model_wrapper.info(verbose=False, imgsz=imgsz)
        # Try common patterns
        if isinstance(info, dict):
            # e.g., info.get("GFLOPs") or nested
            if "GFLOPs" in info:
                gflops = float(info["GFLOPs"])
            elif "flops" in info:
                # often in MFLOPs or GFLOPs; assume GFLOPs if large
                gflops = float(info["flops"])
        # If info is something else (string, None), we leave gflops as NaN
    except TypeError:
        # Older versions may not accept imgsz as kwarg
        try:
            info = model_wrapper.info(verbose=False)
            if isinstance(info, dict) and "GFLOPs" in info:
                gflops = float(info["GFLOPs"])
        except Exception:
            pass
    except Exception:
        pass

    if gflops != gflops:  # NaN check
        print("[WARN] Could not extract GFLOPs programmatically from model.info(); "
              "will report 'NaN' in CSV. Check Ultralytics version if needed.")

    # -----------------------------
    # FPS measurement
    # -----------------------------
    bs = 1
    dummy = torch.randn(bs, 3, imgsz, imgsz, device=device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    avg_time = total_time / runs
    ms_per_img = avg_time * 1000.0
    fps = 1.0 / avg_time if avg_time > 0 else float("inf")

    print("  Params      : {:.3f} M ({:,})".format(n_params_m, n_params))
    print("  Layers      : {}".format(n_layers))
    print("  GFLOPs      : {}".format("NaN" if gflops != gflops else f"{gflops:.3f}"))
    print("  Time / img  : {:.3f} ms".format(ms_per_img))
    print("  FPS         : {:.2f}".format(fps))

    # Return dict for CSV
    return {
        "model_name": weights_path.stem,
        "weights_path": str(weights_path),
        "params": n_params,
        "params_M": n_params_m,
        "layers": n_layers,
        "GFLOPs": gflops,
        "imgsz": imgsz,
        "runs": runs,
        "warmup": warmup,
        "ms_per_image": ms_per_img,
        "FPS": fps,
        "device": str(device),
    }


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
def main():
    args = parse_args()
    device = select_device(args.device)

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    model_paths = sorted(p for p in folder.glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No .pt files found in {folder}")

    print(f"[INFO] Found {len(model_paths)} model(s) in {folder}")

    # Prepare CSV
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "weights_path",
        "params",
        "params_M",
        "layers",
        "GFLOPs",
        "imgsz",
        "runs",
        "warmup",
        "ms_per_image",
        "FPS",
        "device",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for wpath in model_paths:
            try:
                row = profile_single_model(
                    weights_path=wpath,
                    imgsz=args.imgsz,
                    device=device,
                    runs=args.runs,
                    warmup=args.warmup,
                )
                writer.writerow(row)
            except Exception as e:
                print(f"[ERROR] Failed to profile {wpath.name}: {e}")

    print(f"\n[INFO] Saved model profiles to: {csv_path}")


if __name__ == "__main__":
    main()


import os
import time
import torch
import pandas as pd
from thop import profile
from pathlib import Path
from ultralytics import YOLO

# Configuration
WEIGHT_DIR = "/home/imad/Documents/models"
INPUT_SHAPE = (3, 640, 640)

def analyze_weights(folder):
    weight_files = list(Path(folder).glob("*.pt"))
    if not weight_files:
        print(f"[⚠️] No .pt files found in {folder}")
        return

    print(f"📁 Found {len(weight_files)} weight file(s)")
    summary = []

    for pt_file in weight_files:
        print(f"\n🔍 Loading: {pt_file.name}")
        try:
            model_wrapper = YOLO(str(pt_file))
            model = model_wrapper.model
            model.eval()

            dummy_input = torch.randn(1, *INPUT_SHAPE)

            # Inference time
            start = time.time()
            with torch.no_grad():
                model(dummy_input)
            elapsed = time.time() - start
            fps = round(1 / elapsed, 2)

            # FLOPs and Params
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

            # Model size (MB)
            size_mb = round(os.path.getsize(pt_file) / (1024 ** 2), 2)

            # Layer count
            layer_count = len(list(model.modules()))

            summary.append({
                "Model": pt_file.stem,
                "Type": model.__class__.__name__,
                "Params (M)": round(params / 1e6, 3),
                "FLOPs (G)": round(flops / 1e9, 3),
                "Size (MB)": size_mb,
                "Layers": layer_count,
                "Inference (s)": round(elapsed, 4),
                "FPS": fps
            })

        except Exception as e:
            print(f"[❌] Failed to process {pt_file.name}: {e}")
            summary.append({
                "Model": pt_file.stem,
                "Type": "Error",
                "Params (M)": "Error",
                "FLOPs (G)": "Error",
                "Size (MB)": "Error",
                "Layers": "Error",
                "Inference (s)": "Error",
                "FPS": "Error"
            })

    # Export results
    df = pd.DataFrame(summary)
    df.to_csv("model_complexity_summary.csv", index=False)
    print("\n✅ Saved to model_complexity_summary.csv")
    print(df)

# Entry
if __name__ == "__main__":
    analyze_weights(WEIGHT_DIR)


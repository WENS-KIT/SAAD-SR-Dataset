import os

# Path to your scale_dataset directory
BASE_DIR = "/home/imad/ultralytics/ultralytics/SAAD_SR_dataset/scale_dataset"

# Altitude folders
altitudes = ["15m", "25m", "45m"]

# Counters
total_train = 0
total_val = 0

print("Counting samples per altitude...\n")

for alt in altitudes:
    alt_path = os.path.join(BASE_DIR, alt)

    train_dir = os.path.join(alt_path, "train", "images")
    val_dir = os.path.join(alt_path, "val", "images")

    # Count files
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # Update totals
    total_train += train_count
    total_val += val_count

    print(f"Altitude {alt}:")
    print(f"  Training samples:   {train_count}")
    print(f"  Validation samples: {val_count}\n")

print("===== Combined Totals =====")
print(f"Total training samples:   {total_train}")
print(f"Total validation samples: {total_val}")


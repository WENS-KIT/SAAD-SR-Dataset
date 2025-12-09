import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class AltitudeDataset(Dataset):
    """
    Altitude-aware dataset loader with integrity check.
    Handles empty/missing/corrupted label files, and prints clean stats.
    """

    def __init__(self, root_dir, altitude_map, img_size=640, split='train'):
        self.samples = []
        self.img_size = img_size
        self.altitude_map = altitude_map
        self.split = split.lower()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        total_images = 0
        missing_label_count = 0
        background_count = 0
        corrupted_label_count = 0
        valid_label_count = 0

        print(f"[INFO] Loading dataset from: {root_dir} | split: {self.split}")
        for alt_folder, altitude in self.altitude_map.items():
            img_dir = os.path.join(root_dir, alt_folder, self.split, 'images')
            label_dir = os.path.join(root_dir, alt_folder, self.split, 'labels')

            print(f"  Checking altitude folder: {alt_folder}")
            if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                print(f"  Skipping - missing directory: {img_dir} or {label_dir}")
                continue

            image_paths = []
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

            for img_path in tqdm(sorted(image_paths), desc=f"  Verifying {alt_folder}"):
                total_images += 1
                basename = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, basename + ".txt")

                if not os.path.exists(label_path):
                    missing_label_count += 1
                    self.samples.append((img_path, None, altitude))
                    continue

                try:
                    with open(label_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        valid_lines = [line for line in lines if len(line.split()) >= 5]

                    if len(lines) != len(valid_lines):
                        corrupted_label_count += 1
                        continue

                    if len(valid_lines) == 0:
                        background_count += 1
                    else:
                        valid_label_count += 1

                    self.samples.append((img_path, label_path, altitude))
                except Exception:
                    corrupted_label_count += 1
                    continue

        print(f"[SUMMARY] Dataset split: '{self.split}'")
        print(f"  Total images found           : {total_images}")
        print(f"  → With valid object labels   : {valid_label_count}")
        print(f"  → Background (empty labels)  : {background_count}")
        print(f"  → Missing label files        : {missing_label_count}")
        print(f"  → Corrupted label files      : {corrupted_label_count}")
        print(f"  → Final valid samples loaded : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, altitude = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label_tensor = torch.zeros((0, 6))
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]
                valid = [list(map(float, l)) for l in lines if len(l) >= 5]
                if valid:
                    labels = [[0] + l for l in valid]
                    label_tensor = torch.tensor(labels, dtype=torch.float32)

        altitude_tensor = torch.tensor([altitude], dtype=torch.float32)
        return image, label_tensor, altitude_tensor


import os
import cv2
from ultralytics import YOLO

def extract_and_label_frames(video_path, image_dir, label_dir, model, start_id=1, conf=0.25, class_map=None):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[🎥] {os.path.basename(video_path)}: {total_frames} frames at {video_fps:.2f} FPS")

    current_id = start_id
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        image_filename = f"{current_id:07d}.jpg"
        label_filename = f"{current_id:07d}.txt"
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, label_filename)

        cv2.imwrite(image_path, frame)
        print(f"[📸] Saved frame: {image_path}")

        height, width = frame.shape[:2]
        results = model(image_path, conf=conf)[0]

        with open(label_path, 'w') as f:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                mapped_cls = class_map.get(cls_id, cls_id) if class_map else cls_id

                x_center = ((x1 + x2) / 2.0) / width
                y_center = ((y1 + y2) / 2.0) / height
                w_norm = (x2 - x1) / width
                h_norm = (y2 - y1) / height

                f.write(f"{mapped_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"[📄] Saved label: {label_path}")
        current_id += 1

    cap.release()
    return current_id


# === MAIN ===

root_video_folder = '/home/imad/Research/vid/videos/drone_dataset/dataset'
output_base = '/home/imad/Research/vid/videos/scale_dataset'
model_path = '/home/imad/Research/vid/yolov8_x/runs/train/yolov8x_1088res/weights/best.pt'

model = YOLO(model_path)
class_mapping = None  # Optional: e.g., {0: 0, 1: 1}

saved_id = 1  # Global ID counter

for scale_folder in sorted(os.listdir(root_video_folder)):
    scale_path = os.path.join(root_video_folder, scale_folder)
    if not os.path.isdir(scale_path):
        continue

    print(f"\n🔍 Processing scale level: {scale_folder}")
    image_output_dir = os.path.join(output_base, scale_folder, 'images')
    label_output_dir = os.path.join(output_base, scale_folder, 'labels')

    for video_file in sorted(os.listdir(scale_path)):
        if video_file.lower().endswith('.mp4'):
            video_path = os.path.join(scale_path, video_file)
            print(f"\n[🎞️] Video: {video_file}")
            saved_id = extract_and_label_frames(
                video_path,
                image_output_dir,
                label_output_dir,
                model,
                start_id=saved_id,
                conf=0.25,
                class_map=class_mapping
            )


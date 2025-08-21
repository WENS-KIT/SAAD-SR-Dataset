# SAAD-SR Dataset

This repository contains the official dataset and benchmark utilities for the paper:

**SAAD-SR: A Scale-Aware Aerial Benchmark for Learning Robust Object Representations Under UAV-Induced Scale Shifts**

SAAD-SR is designed to evaluate object detection robustness in aerial scenarios with altitude-induced scale variation and environmental diversity. The dataset includes:

- Discretized UAV telemetry (altitude-stratified views)
- Multi-condition image sequences (day, night, snow, haze)
- Altitude-specific annotations
- Per-class statistics
- Tools for image extraction, label generation, and human-in-the-loop correction

---

## 🔗 Dataset Download

> 📦 **[Link will be provided soon]**

## 💾 Trained Models

> 🧠 **Coming soon**

---

## Repository Structure

```bash
.
├── data.py         # Extract frames from aerial videos
├── annotator.py    # Predict labels using pretrained YOLO/RT-DETR weights
├── gui.py          # GUI to correct or verify labels (LabelMe-style)
├── dataset/        # Contains image frames and predicted/corrected labels
├── models/         # Folder for pretrained weights (optional)
├── README.md       # This file
└── requirements.txt
```
## 🔧 Setup

```bash
python3 -m venv saad-sr-env
source saad-sr-env/bin/activate
pip install -r requirements.txt
```
## 📍 Usage
## 1. Extract Image Frames from Aerial Videos
```bash
python3 data.py --video_dir path/to/videos --out_dir dataset/images --fps 1
```
## 2. Generate Labels with Pretrained Weights
```bash
python3 annotator.py --image_dir dataset/images --out_dir dataset/labels --weights models/yolov8-sr.pt

```

## Launches an interactive labeling tool:

 -  View predictions

-   Adjust/add bounding boxes

-  Save corrected annotations


## 3. Launch GUI for Label Correction
```bash
python3 gui.py 

```
Launches an interactive labeling tool:

 View predictions

Adjust/add bounding boxes

 Save corrected annotations

🛰️ Designed for efficient correction across altitude variations and weather conditions

📊 Benchmark Protocols

Evaluation scripts and trained models will be released shortly. The dataset supports:

 Altitude-aware detection

 Weather-conditioned generalization

 Scale Robustness Index (SRI)


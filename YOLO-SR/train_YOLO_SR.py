import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics.models.detection_model_sr import DetectionModelSR
from ultralytics.altitude_dataset import AltitudeDataset
from ultralytics.nn.tasks import v8DetectionLoss
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from collections import defaultdict
import torch.nn.functional as F
import yaml
import requests
from urllib.parse import urlparse



# =========================
# CONFIG
# =========================
DATA_DIR = Path("/home/imad/ultralytics/ultralytics/SAAD_SR_dataset/sample_dataset")
ALTITUDE_MAP = {"15m": 15.0, "25m": 25.0, "45m": 45.0}
IMAGE_SIZE = 640

BATCH_SIZE = 8
ACCUMULATE_STEPS = 8
NUM_EPOCHS = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_WEIGHTS = "yolov8n.pt"
FREEZE_BACKBONE_EPOCHS = 1  # unfreeze early

USE_AMP = True

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]
NUM_CLASSES = len(CLASS_NAMES)

WEIGHT_DIR = Path("run/weights")
PLOT_DIR = Path("run/plots")
LOG_DIR = Path("run/logs")
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)


# =========================
# UTILS
# =========================
def download_pretrained_weights(model_name: str):
    weights_path = Path(model_name)
    if weights_path.exists():
        print(f" Found existing weights: {weights_path}")
        return str(weights_path)

    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
    }

    if model_name not in model_urls:
        print(f" Unknown model: {model_name}")
        return None

    print(f" Downloading {model_name}...")
    try:
        resp = requests.get(model_urls[model_name], stream=True)
        resp.raise_for_status()
        with open(weights_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f" Downloaded: {weights_path}")
        return str(weights_path)
    except Exception as e:
        print(f" Failed to download {model_name}: {e}")
        return None


def load_pretrained_weights(model, weights_path):
    if not weights_path or not Path(weights_path).exists():
        print(" No pretrained weights found, training from scratch")
        return model

    try:
        print(f" Loading BACKBONE weights only from: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

        model_dict = model.state_dict()
        backbone_dict = {}
        head_dict = {}

        for k, v in state_dict.items():
            is_detection_head = any(p in k for p in [
                "model.22", "model.23", "model.24",
                "head", "detect", "dfl", "cv2.2", "cv3.2"
            ])
            if is_detection_head:
                head_dict[k] = v
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                backbone_dict[k] = v
            elif k in model_dict:
                print(f"  Skipping backbone layer {k}: shape mismatch ({model_dict[k].shape} vs {v.shape})")

        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
        print(f" Transfer Learning: Loaded {len(backbone_dict)} backbone layers (skipped {len(head_dict)} head layers)")
        return model
    except Exception as e:
        print(f" Error loading pretrained weights: {e}")
        print(" Continuing with random initialization")
        return model


def freeze_backbone(model, freeze=True):
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        is_detection_head = any(s in name for s in [
            "model.22", "model.23", "model.24",
            "head", "detect", "dfl", "cv2.2", "cv3.2"
        ])
        if freeze:
            if is_detection_head:
                p.requires_grad = True
                trainable += 1
            else:
                p.requires_grad = False
                frozen += 1
        else:
            p.requires_grad = True
            trainable += 1
    if freeze:
        print(f" Transfer Learning: Frozen {frozen} backbone params, training {trainable} head params")
    else:
        print(f"Full training: {trainable} trainable params")
    return trainable > 0


def custom_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    altitudes = torch.stack([b[2] for b in batch])
    return images, labels, altitudes


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x.transpose(0, -1)[xc[xi]]

        if not x.shape[0]:
            continue

        box, cls = x[:, :4], x[:, 4:mi]
        box = xywh2xyxy(box)

        if nc > 1:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], cls[i, j].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        boxes, scores = x[:, :4], x[:, 4]
        keep = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]
        output[xi] = x[keep]

    return output


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# =========================
# EVALUATION
# =========================
def evaluate_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.25):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels, altitudes in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            altitudes = altitudes.to(device)

            try:
                with torch.amp.autocast("cuda", enabled=USE_AMP and torch.cuda.is_available()):
                    preds, _ = model(images, altitude=altitudes)

                # flatten list of feature maps
                if isinstance(preds, list):
                    preds = torch.cat([p.flatten(2) for p in preds], dim=2)

                pred_nms = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=0.45)

                for pred, target_labels in zip(pred_nms, labels):
                    if target_labels.numel() == 0:
                        continue
                    targets = target_labels[:, 1:].clone()
                    if targets.numel() > 0:
                        targets[:, 1:] = xywh2xyxy(targets[:, 1:])
                    all_predictions.append(pred)
                    all_targets.append(targets)

            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    if not all_predictions:
        return 0.0, {}, {}

    aps = []
    class_metrics = defaultdict(lambda: {"precision": 0, "recall": 0, "ap": 0})

    for class_id in range(NUM_CLASSES):
        class_preds = []
        class_targets = []
        for pred, target in zip(all_predictions, all_targets):
            if pred.numel() > 0:
                c_pred = pred[pred[:, 5] == class_id]
                class_preds.extend(c_pred.cpu().numpy())
            if target.numel() > 0:
                c_tgt = target[target[:, 0] == class_id]
                class_targets.extend(c_tgt.cpu().numpy())

        if not class_preds and not class_targets:
            continue
        if not class_preds:
            class_metrics[class_id] = {"precision": 0, "recall": 0, "ap": 0}
            continue
        if not class_targets:
            class_metrics[class_id] = {"precision": 0, "recall": 1, "ap": 0}
            continue

        class_preds = np.array(class_preds)
        class_targets = np.array(class_targets)

        order = np.argsort(-class_preds[:, 4])
        class_preds = class_preds[order]

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            pb = pred[:4]
            max_iou = 0
            for tgt in class_targets:
                tb = tgt[1:5]
                inter = max(0, min(pb[2], tb[2]) - max(pb[0], tb[0])) * \
                        max(0, min(pb[3], tb[3]) - max(pb[1], tb[1]))
                area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                area_t = (tb[2] - tb[0]) * (tb[3] - tb[1])
                union = area_p + area_t - inter
                iou = inter / union if union > 0 else 0
                max_iou = max(max_iou, iou)
            if max_iou >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / len(class_targets)
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        ap = compute_ap(recall, precision)

        class_metrics[class_id] = {
            "precision": precision[-1] if len(precision) else 0,
            "recall": recall[-1] if len(recall) else 0,
            "ap": ap
        }
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0
    return mAP, dict(class_metrics), {}


# =========================
# DATA LOADERS
# =========================
print(" Preparing TRAIN loader")
train_dataset = AltitudeDataset(DATA_DIR, ALTITUDE_MAP, IMAGE_SIZE, split="train")
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    collate_fn=custom_collate_fn, pin_memory=True
)

print(" Preparing VAL loader")
val_dataset = AltitudeDataset(DATA_DIR, ALTITUDE_MAP, IMAGE_SIZE, split="val")
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    collate_fn=custom_collate_fn, pin_memory=True
)

# =========================
# MODEL + OPTIM
# =========================
print(" Initializing model...")
model = DetectionModelSR(
    cfg="ultralytics/cfg/models/v8/yolov8sr.yaml",
    nc=NUM_CLASSES,
    ch=3
).to(DEVICE)

# lower cls weight
model.args = type("OBJ", (), {})()
model.args.box = 0.05
model.args.cls = 0.05
model.args.dfl = 1.5
model.args.hyp = {"box": 0.05, "cls": 0.05, "dfl": 1.5}

print(f"[INFO] DetectionModelSR initialized with {NUM_CLASSES} classes")

weights_path = download_pretrained_weights(PRETRAINED_WEIGHTS)
if weights_path:
    model = load_pretrained_weights(model, weights_path)

loss_fn = v8DetectionLoss(model)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8
)

total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATE_STEPS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=25,
    final_div_factor=1e4
)

if torch.cuda.is_available():
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
else:
    scaler = torch.amp.GradScaler("cpu", enabled=False)


def wrap_labels(label_list):
    if not label_list or all(l.numel() == 0 for l in label_list):
        return {"batch_idx": torch.empty(0), "cls": torch.empty(0), "bboxes": torch.empty(0, 4)}
    all_labels = torch.cat([l for l in label_list if l.numel() > 0], 0).cpu()
    return {
        "batch_idx": all_labels[:, 0].long(),
        "cls": all_labels[:, 1].long(),
        "bboxes": all_labels[:, 2:6],
    }


def plot_metrics(metrics, out_dir):
    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    def _plot(name, train_vals, val_vals=None):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_vals, label=f"Train {name}", linewidth=2)
        if val_vals:
            plt.plot(epochs, val_vals, label=f"Val {name}", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name} over epochs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close()

    _plot("Loss", metrics["train_loss"], metrics["val_loss"])
    _plot("Box Loss", metrics["train_box"], metrics["val_box"])
    _plot("Cls Loss", metrics["train_cls"], metrics["val_cls"])
    _plot("DFL Loss", metrics["train_dfl"], metrics["val_dfl"])
    _plot("mAP@0.5", metrics["val_map"])
    _plot("Learning Rate", metrics["learning_rate"])


# =========================
# TRAIN LOOP
# =========================
best_map = 0.0
metrics = {
    "train_loss": [], "val_loss": [],
    "train_box": [], "val_box": [],
    "train_cls": [], "val_cls": [],
    "train_dfl": [], "val_dfl": [],
    "val_map": [], "val_precision": [], "val_recall": [],
    "learning_rate": []
}

print(f"\n{'Epoch':<8}{'GPU_mem':>10}{'box_loss':>10}{'cls_loss':>10}{'dfl_loss':>10}{'mAP@0.5':>10}{'Precision':>12}{'Recall':>10}{'LR':>12}")

for epoch in range(NUM_EPOCHS):
    # freeze/unfreeze
    if epoch == 0:
        ok = freeze_backbone(model, freeze=True)
        if not ok:
            freeze_backbone(model, freeze=False)
    elif epoch == FREEZE_BACKBONE_EPOCHS:
        freeze_backbone(model, freeze=False)
        print(f"Switching to full model training at epoch {epoch+1}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    train_loss = train_box = train_cls = train_dfl = 0.0
    train_steps = 0
    train_bar = tqdm(train_loader, desc=f"[Train {epoch+1:>3}/{NUM_EPOCHS}]", dynamic_ncols=True)
    optimizer.zero_grad()

    for batch_idx, (images, labels, altitudes) in enumerate(train_bar):
        images = images.to(DEVICE, non_blocking=True)
        altitudes = altitudes.to(DEVICE, non_blocking=True)
        label_dict = wrap_labels(labels)

        try:
            with torch.amp.autocast("cuda", enabled=USE_AMP and torch.cuda.is_available()):
                preds, feats = model(images, altitude=altitudes)
                if preds is None:
                    continue
                loss, loss_items = loss_fn(preds, label_dict, feats=feats)
                loss = loss / ACCUMULATE_STEPS

            scaler.scale(loss).backward()

            # flag to track whether we actually stepped optimizer
            did_update = False

            # accumulation boundary
            if (batch_idx + 1) % ACCUMULATE_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # unscale and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                did_update = True

            # only step scheduler if optimizer actually stepped
            if did_update:
                scheduler.step()

            # logging
            box = loss_items[0].item()
            cls = loss_items[1].item()
            dfl = loss_items[2].item()

            train_loss += loss.item() * ACCUMULATE_STEPS
            train_box += box
            train_cls += cls
            train_dfl += dfl
            train_steps += 1

            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]["lr"]
            train_bar.set_postfix({
                "GPU_mem": f"{gpu_mem:.1f}G",
                "box_loss": f"{box:.4f}",
                "cls_loss": f"{cls:.4f}",
                "dfl_loss": f"{dfl:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        except Exception as e:
            print(f"Training error: {e}")
            continue

    # ====== VALIDATION ======
    model.eval()
    val_loss = val_box = val_cls = val_dfl = 0.0
    val_steps = 0
    val_bar = tqdm(val_loader, desc="[Val]  ", dynamic_ncols=True)
    with torch.no_grad():
        for images, labels, altitudes in val_bar:
            images = images.to(DEVICE, non_blocking=True)
            altitudes = altitudes.to(DEVICE, non_blocking=True)
            label_dict = wrap_labels(labels)
            try:
                with torch.amp.autocast("cuda", enabled=USE_AMP and torch.cuda.is_available()):
                    preds, feats = model(images, altitude=altitudes)
                    if preds is None:
                        continue
                    loss, loss_items = loss_fn(preds, label_dict, feats=feats)

                val_loss += loss.item()
                val_box += loss_items[0].item()
                val_cls += loss_items[1].item()
                val_dfl += loss_items[2].item()
                val_steps += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    print("\n Computing mAP and class metrics...")
    val_map, class_metrics, _ = evaluate_model(model, val_loader, DEVICE)
    avg_precision = np.mean([m["precision"] for m in class_metrics.values()]) if class_metrics else 0
    avg_recall = np.mean([m["recall"] for m in class_metrics.values()]) if class_metrics else 0

    avg_val_loss = val_loss / max(val_steps, 1)
    avg_val_box = val_box / max(val_steps, 1)
    avg_val_cls = val_cls / max(val_steps, 1)
    avg_val_dfl = val_dfl / max(val_steps, 1)
    current_lr = optimizer.param_groups[0]["lr"]

    metrics["train_loss"].append(train_loss / max(train_steps, 1))
    metrics["val_loss"].append(avg_val_loss)
    metrics["train_box"].append(train_box / max(train_steps, 1))
    metrics["val_box"].append(avg_val_box)
    metrics["train_cls"].append(train_cls / max(train_steps, 1))
    metrics["val_cls"].append(avg_val_cls)
    metrics["train_dfl"].append(train_dfl / max(train_steps, 1))
    metrics["val_dfl"].append(avg_val_dfl)
    metrics["val_map"].append(val_map)
    metrics["val_precision"].append(avg_precision)
    metrics["val_recall"].append(avg_recall)
    metrics["learning_rate"].append(current_lr)

    writer.add_scalar("Validation/Loss", avg_val_loss, epoch)
    writer.add_scalar("Validation/mAP@0.5", val_map, epoch)
    writer.add_scalar("Validation/Precision", avg_precision, epoch)
    writer.add_scalar("Validation/Recall", avg_recall, epoch)

    gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"{epoch+1:>3}/{NUM_EPOCHS:<4} {gpu_mem:10.1f}G {avg_val_box:10.4f}{avg_val_cls:10.4f}{avg_val_dfl:10.4f}{val_map:10.4f}{avg_precision:12.4f}{avg_recall:10.4f}{current_lr:12.2e}")

    # save checkpoints
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map": best_map,
        "metrics": metrics
    }, WEIGHT_DIR / "last.pt")

    if val_map > best_map:
        best_map = val_map
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "metrics": metrics
        }, WEIGHT_DIR / "best.pt")
        print(f" Saved best model — mAP@0.5: {val_map:.4f}")

# exports
CSV_PATH = Path("run/results.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_PATH, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    header = [
        "epoch", "train_loss", "val_loss",
        "train_box", "val_box", "train_cls", "val_cls",
        "train_dfl", "val_dfl", "mAP50", "avg_precision", "avg_recall",
        "learning_rate", "gpu_memory_gb"
    ]
    for class_name in CLASS_NAMES:
        header.extend([f"{class_name}_precision", f"{class_name}_recall", f"{class_name}_ap"])
    writer_csv.writerow(header)

    for i in range(len(metrics["train_loss"])):
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        row = [
            i + 1,
            round(metrics["train_loss"][i], 6),
            round(metrics["val_loss"][i], 6),
            round(metrics["train_box"][i], 6),
            round(metrics["val_box"][i], 6),
            round(metrics["train_cls"][i], 6),
            round(metrics["val_cls"][i], 6),
            round(metrics["train_dfl"][i], 6),
            round(metrics["val_dfl"][i], 6),
            round(metrics["val_map"][i], 6),
            round(metrics["val_precision"][i], 6),
            round(metrics["val_recall"][i], 6),
            f"{metrics['learning_rate'][i]:.2e}",
            round(gpu_mem, 2)
        ]
        for _ in CLASS_NAMES:
            row.extend([0.0, 0.0, 0.0])
        writer_csv.writerow(row)

config_path = Path("run/training_config.yaml")
training_config = {
    'model': {
        'pretrained_weights': PRETRAINED_WEIGHTS,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'image_size': IMAGE_SIZE
    },
    'training': {
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'accumulate_steps': ACCUMULATE_STEPS,
        'effective_batch_size': BATCH_SIZE * ACCUMULATE_STEPS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'freeze_backbone_epochs': FREEZE_BACKBONE_EPOCHS
    },
    'results': {
        'best_map': float(best_map),
        'final_lr': float(optimizer.param_groups[0]['lr']),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
}
with open(config_path, 'w') as f:
    yaml.dump(training_config, f, default_flow_style=False, indent=2)

plot_metrics(metrics, PLOT_DIR)
writer.close()

print("\nDone.")

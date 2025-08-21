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

# CONFIG
DATA_DIR = Path("/home/imad/ultralytics/ultralytics/scale_dataset")
ALTITUDE_MAP = {"15m": 15.0, "25m": 25.0, "45m": 45.0}
IMAGE_SIZE = 640
BATCH_SIZE = 2  # Physical batch size
ACCUMULATE_STEPS = 8  # Gradient accumulation steps (effective batch size = BATCH_SIZE * ACCUMULATE_STEPS = 16)
NUM_EPOCHS = 100  # Increased for OneCycleLR
LEARNING_RATE = 1e-3  # Higher initial LR for OneCycleLR
WEIGHT_DECAY = 1e-4  # For AdamW
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer Learning Configuration
PRETRAINED_WEIGHTS = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
FREEZE_BACKBONE_EPOCHS = 10  # Number of epochs to freeze backbone for transfer learning

# Mixed Precision Training
USE_AMP = True  # Enable Automatic Mixed Precision

# Class names (update according to your dataset)
CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
NUM_CLASSES = len(CLASS_NAMES)

WEIGHT_DIR = Path("run/weights")
PLOT_DIR = Path("run/plots")
LOG_DIR = Path("run/logs")
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=LOG_DIR)

def download_pretrained_weights(model_name):
    """Download pretrained YOLOv8 weights if not available locally."""
    weights_path = Path(model_name)
    if weights_path.exists():
        print(f" Found existing weights: {weights_path}")
        return str(weights_path)
    
    # YOLOv8 model URLs
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
        response = requests.get(model_urls[model_name], stream=True)
        response.raise_for_status()
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f" Downloaded: {weights_path}")
        return str(weights_path)
    except Exception as e:
        print(f" Failed to download {model_name}: {e}")
        return None

def load_pretrained_weights(model, weights_path):
    """Load pretrained weights - BACKBONE ONLY for transfer learning."""
    if not weights_path or not Path(weights_path).exists():
        print(" No pretrained weights found, training from scratch")
        return model
    
    try:
        print(f" Loading BACKBONE weights only from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Filter to ONLY backbone layers - exclude detection head
        model_dict = model.state_dict()
        backbone_dict = {}
        head_dict = {}
        
        for k, v in state_dict.items():
            # Skip detection head layers completely
            is_detection_head = any(pattern in k for pattern in [
                'model.22',  # Standard YOLO detection layers
                'model.23', 
                'model.24',
                'head',      # Generic head patterns
                'detect',
                'cv2.2',     # Classification layers in detection head
                'cv3.2',     # Box regression layers in detection head
                'dfl'        # Distribution Focal Loss layers
            ])
            
            if is_detection_head:
                head_dict[k] = v
                continue
                
            # Only load backbone weights that match in shape
            if k in model_dict and model_dict[k].shape == v.shape:
                backbone_dict[k] = v
            elif k in model_dict:
                print(f"  Skipping backbone layer {k}: shape mismatch ({model_dict[k].shape} vs {v.shape})")
        
        # Load only the backbone weights
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
        
        print(f" Transfer Learning: Loaded {len(backbone_dict)} backbone layers (skipped {len(head_dict)} detection head layers)")
        print(f" Detection head will be trained from scratch with your custom architecture")
        return model
        
    except Exception as e:
        print(f" Error loading pretrained weights: {e}")
        print(" Continuing with random initialization")
        return model

def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone layers for transfer learning."""
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Define what constitutes the detection head in your custom model
        is_detection_head = any(pattern in name for pattern in [
            'model.22',  # Standard YOLO detection layers
            'model.23', 
            'model.24',
            'head',      # Generic head patterns
            'detect',
            'cv2.2',     # Classification layers in detection head
            'cv3.2',     # Box regression layers in detection head  
            'dfl'        # Distribution Focal Loss layers
        ])
        
        if freeze:
            if is_detection_head:
                # Always keep detection head trainable (since it's your custom head)
                param.requires_grad = True
                trainable_count += 1
            else:
                # Freeze backbone layers (these have pretrained weights)
                param.requires_grad = False
                frozen_count += 1
        else:
            # Unfreeze everything for full training
            param.requires_grad = True
            trainable_count += 1
    
    if freeze:
        print(f" Transfer Learning: Frozen {frozen_count} backbone parameters (pretrained)")
        print(f" Training {trainable_count} detection head parameters (custom architecture)")
    else:
        print(f"Full training: All {trainable_count} parameters unfrozen")
    
    return trainable_count > 0  # Return True if we have trainable parameters

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    altitudes = torch.stack([item[2] for item in batch])
    return images, labels, altitudes

def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Apply Non-Maximum Suppression (NMS) to predictions."""
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = x.split((4, nc), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if not merge else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output

def compute_ap(recall, precision):
    """Compute Average Precision (AP) given precision and recall curves."""
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Look for points where x-axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.25):
    """Evaluate model and compute mAP, precision, recall metrics."""
    model.eval()
    stats = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, altitudes in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            altitudes = altitudes.to(device)
            
            # Get predictions
            try:
                with torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
                    preds, _ = model(images, altitude=altitudes)
                    if preds is None:
                        continue
                        
                # Apply NMS
                pred_nms = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=0.45)
                
                # Process each image in batch
                for i, (pred, target_labels) in enumerate(zip(pred_nms, labels)):
                    if target_labels.numel() == 0:
                        continue
                        
                    # Convert target format (batch_idx, class, x, y, w, h) to (class, x1, y1, x2, y2)
                    targets = target_labels[:, 1:].clone()  # Remove batch_idx
                    if targets.numel() > 0:
                        targets[:, 1:] = xywh2xyxy(targets[:, 1:])  # Convert xywh to xyxy
                    
                    all_predictions.append(pred)
                    all_targets.append(targets)
                    
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Compute mAP and class-wise metrics
    if not all_predictions:
        return 0.0, {}, {}
    
    # Combine all predictions and targets
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # For mAP@0.5:0.95
    aps = []
    class_metrics = defaultdict(lambda: {'precision': 0, 'recall': 0, 'ap': 0})
    
    for class_id in range(NUM_CLASSES):
        # Get predictions and targets for this class
        class_preds = []
        class_targets = []
        
        for pred, target in zip(all_predictions, all_targets):
            if pred.numel() > 0:
                class_pred = pred[pred[:, 5] == class_id]  # Filter by class
                class_preds.extend(class_pred.cpu().numpy())
            
            if target.numel() > 0:
                class_target = target[target[:, 0] == class_id]  # Filter by class
                class_targets.extend(class_target.cpu().numpy())
        
        if not class_preds and not class_targets:
            continue
            
        if not class_preds:
            class_metrics[class_id] = {'precision': 0, 'recall': 0, 'ap': 0}
            continue
            
        if not class_targets:
            class_metrics[class_id] = {'precision': 0, 'recall': 1, 'ap': 0}
            continue
        
        # Convert to numpy arrays
        class_preds = np.array(class_preds)
        class_targets = np.array(class_targets)
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(-class_preds[:, 4])
        class_preds = class_preds[sorted_indices]
        
        # Compute precision and recall
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for i, pred in enumerate(class_preds):
            pred_box = pred[:4]
            max_iou = 0
            
            for target in class_targets:
                target_box = target[1:5]
                # Compute IoU
                intersection = max(0, min(pred_box[2], target_box[2]) - max(pred_box[0], target_box[0])) * \
                              max(0, min(pred_box[3], target_box[3]) - max(pred_box[1], target_box[1]))
                area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                area_target = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                union = area_pred + area_target - intersection
                iou = intersection / union if union > 0 else 0
                max_iou = max(max_iou, iou)
            
            if max_iou >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(class_targets) if len(class_targets) > 0 else np.zeros_like(tp_cumsum)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # Compute AP
        ap = compute_ap(recall, precision)
        
        # Store class metrics
        final_precision = precision[-1] if len(precision) > 0 else 0
        final_recall = recall[-1] if len(recall) > 0 else 0
        
        class_metrics[class_id] = {
            'precision': final_precision,
            'recall': final_recall,
            'ap': ap
        }
        aps.append(ap)
    
    # Compute mean AP
    mAP = np.mean(aps) if aps else 0.0
    
    return mAP, dict(class_metrics), {}

# Dataset
print(" Preparing TRAIN loader")
train_dataset = AltitudeDataset(DATA_DIR, ALTITUDE_MAP, IMAGE_SIZE, split='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, 
                         collate_fn=custom_collate_fn, pin_memory=True)

print(" Preparing VAL loader")
val_dataset = AltitudeDataset(DATA_DIR, ALTITUDE_MAP, IMAGE_SIZE, split='val')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                       collate_fn=custom_collate_fn, pin_memory=True)

# Model with Transfer Learning
print(" Initializing model...")
model = DetectionModelSR(cfg="ultralytics/cfg/models/v8/yolov8sr.yaml", nc=NUM_CLASSES, ch=3).to(DEVICE)
model.args = type("OBJ", (), {})()
model.args.box = 0.05
model.args.cls = 0.5
model.args.dfl = 1.5
model.args.hyp = {"box": 0.05, "cls": 0.5, "dfl": 1.5}

def debug_model_structure(model, save_to_file=True):
    """Debug function to understand model structure and parameter names."""
    print("\n Model Structure Analysis:")
    
    all_params = list(model.named_parameters())
    print(f"   Total parameters: {len(all_params)}")
    
    # Group parameters by layer prefix
    layer_groups = {}
    for name, param in all_params:
        if 'model.' in name:
            try:
                layer_prefix = '.'.join(name.split('.')[:2])  # e.g., 'model.0', 'model.22'
                if layer_prefix not in layer_groups:
                    layer_groups[layer_prefix] = []
                layer_groups[layer_prefix].append((name, param.shape, param.numel()))
            except:
                continue
    
    # Print layer summary
    print(f"   Found {len(layer_groups)} layer groups:")
    for layer_prefix in sorted(layer_groups.keys(), key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 999):
        params_in_layer = len(layer_groups[layer_prefix])
        total_params = sum(p[2] for p in layer_groups[layer_prefix])
        print(f"     {layer_prefix}: {params_in_layer} parameters ({total_params:,} total params)")
    
    # Show the last few layers (likely detection head)
    print(f"\n   Last 5 layer groups (likely detection head):")
    last_layers = sorted(layer_groups.keys(), key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 999)[-5:]
    for layer_prefix in last_layers:
        example_param = layer_groups[layer_prefix][0]
        print(f"     {layer_prefix}: {example_param[0]} (shape: {example_param[1]})")
    
    if save_to_file:
        # Save detailed structure to file
        with open("run/model_structure.txt", "w") as f:
            f.write("Model Structure - All Parameters:\n")
            f.write("="*50 + "\n")
            for name, param in all_params:
                f.write(f"{name}: {param.shape} ({param.numel():,} params)\n")
        print(f" Saved detailed structure to: run/model_structure.txt")

# Load pretrained weights
weights_path = download_pretrained_weights(PRETRAINED_WEIGHTS)
if weights_path:
    model = load_pretrained_weights(model, weights_path)

# Debug model structure to understand layer names
debug_model_structure(model)

# Initialize training components
loss_fn = v8DetectionLoss(model)

# AdamW Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                       betas=(0.9, 0.999), eps=1e-8)

# OneCycleLR Scheduler
total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATE_STEPS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE,
    total_steps=total_steps,
    pct_start=0.1,  # 10% of training for warmup
    anneal_strategy='cos',
    div_factor=25,  # initial_lr = max_lr / div_factor
    final_div_factor=1e4  # final_lr = initial_lr / final_div_factor
)

# Mixed Precision Scaler - Updated for newer PyTorch versions
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
else:
    scaler = torch.amp.GradScaler('cpu', enabled=False)  # CPU doesn't support AMP

def wrap_labels(label_list):
    if not label_list or all([l.numel() == 0 for l in label_list]):
        return {"batch_idx": torch.empty(0), "cls": torch.empty(0), "bboxes": torch.empty(0, 4)}
    all_labels = torch.cat([l for l in label_list if l.numel() > 0], 0).cpu()
    return {
        "batch_idx": all_labels[:, 0].long(),
        "cls": all_labels[:, 1].long(),
        "bboxes": all_labels[:, 2:6]
    }

def plot_metrics(metrics, out_dir):
    epochs = list(range(1, len(metrics["train_loss"]) + 1))
    def plot(metric_name, train_vals, val_vals=None):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_vals, label=f'Train {metric_name}', linewidth=2)
        if val_vals:
            plt.plot(epochs, val_vals, label=f'Val {metric_name}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{metric_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    plot("Loss", metrics["train_loss"], metrics["val_loss"])
    plot("Box Loss", metrics["train_box"], metrics["val_box"])
    plot("Cls Loss", metrics["train_cls"], metrics["val_cls"])
    plot("DFL Loss", metrics["train_dfl"], metrics["val_dfl"])
    plot("mAP@0.5", metrics["val_map"])
    plot("Learning Rate", metrics["learning_rate"])

# Training Loop with Enhanced Features
best_val_loss = float('inf')
best_map = 0.0
metrics = {
    "train_loss": [], "val_loss": [],
    "train_box": [], "val_box": [],
    "train_cls": [], "val_cls": [],
    "train_dfl": [], "val_dfl": [],
    "val_map": [], "val_precision": [], "val_recall": [],
    "learning_rate": []
}

print(f"\n Starting enhanced training for {NUM_EPOCHS} epochs...")
print(f"   • Transfer Learning: BACKBONE ONLY from {PRETRAINED_WEIGHTS}")
print(f"   • Custom Detection Head: Training from scratch")
print(f"   • Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"   • Scheduler: OneCycleLR")
print(f"   • Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
print(f"   • Gradient Accumulation: {ACCUMULATE_STEPS} steps (effective batch size: {BATCH_SIZE * ACCUMULATE_STEPS})")
print(f"   • Backbone Freeze: First {FREEZE_BACKBONE_EPOCHS} epochs")

print(f"\n{'Epoch':<8}{'GPU_mem':>10}{'box_loss':>10}{'cls_loss':>10}{'dfl_loss':>10}{'mAP@0.5':>10}{'Precision':>12}{'Recall':>10}{'LR':>12}")

for epoch in range(NUM_EPOCHS):
    # Handle backbone freezing for transfer learning
    if epoch == 0:
        has_trainable = freeze_backbone(model, freeze=True)
        if not has_trainable:
            print(" No trainable parameters found! Starting with full model training instead.")
            freeze_backbone(model, freeze=False)
        else:
            print(" Transfer learning: Frozen backbone (pretrained) + trainable head (custom)")
    elif epoch == FREEZE_BACKBONE_EPOCHS:
        freeze_backbone(model, freeze=False)
        print(f"Switching to full model training at epoch {epoch + 1}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    train_loss = train_box = train_cls = train_dfl = 0.0
    train_steps = 0
    train_bar = tqdm(train_loader, desc=f"[Train {epoch+1:>3}/{NUM_EPOCHS}]", dynamic_ncols=True)
    
    # Reset gradients at start of epoch
    optimizer.zero_grad()

    for batch_idx, (images, labels, altitudes) in enumerate(train_bar):
        images, altitudes = images.to(DEVICE, non_blocking=True), altitudes.to(DEVICE, non_blocking=True)
        label_dict = wrap_labels(labels)
        
        try:
            # Forward pass with mixed precision - Updated for newer PyTorch
            with torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
                preds, feats = model(images, altitude=altitudes)
                if preds is None:
                    continue
                loss, loss_items = loss_fn(preds, label_dict, feats=feats)
                
                # Scale loss for gradient accumulation
                loss = loss / ACCUMULATE_STEPS
                
                if not torch.isfinite(loss).all():
                    continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % ACCUMULATE_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping before optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # OneCycleLR steps every batch
                optimizer.zero_grad()

            # Accumulate losses (multiply back by ACCUMULATE_STEPS for logging)
            box = loss_items[0].item() 
            cls = loss_items[1].item()
            dfl = loss_items[2].item()
            train_loss += loss.item() * ACCUMULATE_STEPS
            train_box += box
            train_cls += cls
            train_dfl += dfl
            train_steps += 1

            # Log to TensorBoard every 50 batches
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('Train/Loss', loss.item() * ACCUMULATE_STEPS, global_step)
                writer.add_scalar('Train/BoxLoss', box, global_step)
                writer.add_scalar('Train/ClsLoss', cls, global_step)
                writer.add_scalar('Train/DFLLoss', dfl, global_step)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                if USE_AMP:
                    writer.add_scalar('Train/GradScale', scaler.get_scale(), global_step)

            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
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

    # Validation
    model.eval()
    val_loss = val_box = val_cls = val_dfl = 0.0
    val_steps = 0
    val_bar = tqdm(val_loader, desc="[Val]  ", dynamic_ncols=True)
    with torch.no_grad():
        for images, labels, altitudes in val_bar:
            images, altitudes = images.to(DEVICE, non_blocking=True), altitudes.to(DEVICE, non_blocking=True)
            label_dict = wrap_labels(labels)
            try:
                with torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
                    preds, feats = model(images, altitude=altitudes)
                    if preds is None:
                        continue
                    loss, loss_items = loss_fn(preds, label_dict, feats=feats)
                    if not torch.isfinite(loss).all():
                        continue
                val_loss += loss.item()
                val_box += loss_items[0].item()
                val_cls += loss_items[1].item()
                val_dfl += loss_items[2].item()
                val_steps += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    # Compute mAP and class-wise metrics
    print("\n Computing mAP and class metrics...")
    val_map, class_metrics, _ = evaluate_model(model, val_loader, DEVICE)
    
    # Compute average precision and recall across all classes
    avg_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()]) if class_metrics else 0
    avg_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()]) if class_metrics else 0

    # Epoch summary
    avg_val_loss = val_loss / max(val_steps, 1)
    avg_val_box = val_box / max(val_steps, 1)
    avg_val_cls = val_cls / max(val_steps, 1)
    avg_val_dfl = val_dfl / max(val_steps, 1)
    current_lr = optimizer.param_groups[0]['lr']

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

    # Log to TensorBoard
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/BoxLoss', avg_val_box, epoch)
    writer.add_scalar('Validation/ClsLoss', avg_val_cls, epoch)
    writer.add_scalar('Validation/DFLLoss', avg_val_dfl, epoch)
    writer.add_scalar('Validation/mAP@0.5', val_map, epoch)
    writer.add_scalar('Validation/Precision', avg_precision, epoch)
    writer.add_scalar('Validation/Recall', avg_recall, epoch)
    writer.add_scalar('Learning_Rate/Epoch', current_lr, epoch)

    # Log class-wise metrics
    for class_id, class_metric in class_metrics.items():
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'class_{class_id}'
        writer.add_scalar(f'ClassMetrics/{class_name}/Precision', class_metric['precision'], epoch)
        writer.add_scalar(f'ClassMetrics/{class_name}/Recall', class_metric['recall'], epoch)
        writer.add_scalar(f'ClassMetrics/{class_name}/AP', class_metric['ap'], epoch)

    gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"{epoch+1:>3}/{NUM_EPOCHS:<4} {gpu_mem:10.1f}G {avg_val_box:10.4f}{avg_val_cls:10.4f}{avg_val_dfl:10.4f}{val_map:10.4f}{avg_precision:12.4f}{avg_recall:10.4f}{current_lr:12.2e}")

    # Save checkpoints
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if USE_AMP else None,
        'best_map': best_map,
        'metrics': metrics
    }, WEIGHT_DIR / "last.pt")
    
    if val_map > best_map:
        best_map = val_map
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if USE_AMP else None,
            'best_map': best_map,
            'metrics': metrics
        }, WEIGHT_DIR / "best.pt")
        print(f" Saved best model — mAP@0.5: {val_map:.4f}")

# Save final training summary
print(f"\n Training Summary:")
print(f"   • Best mAP@0.5: {best_map:.4f}")
print(f"   • Final Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
print(f"   • Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   • Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ---------------- Enhanced CSV Export ----------------

CSV_PATH = Path("run/results.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    
    # Write header with enhanced metrics
    header = [
        "epoch", "train_loss", "val_loss",
        "train_box", "val_box", "train_cls", "val_cls",
        "train_dfl", "val_dfl", "mAP50", "avg_precision", "avg_recall",
        "learning_rate", "gpu_memory_gb"
    ]
    
    # Add class-wise headers
    for class_name in CLASS_NAMES:
        header.extend([f"{class_name}_precision", f"{class_name}_recall", f"{class_name}_ap"])
    
    writer_csv.writerow(header)
    
    # Write enhanced training metrics
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
        
        # Add class-wise metrics for this epoch (placeholder - extend based on your evaluation)
        for class_id in range(len(CLASS_NAMES)):
            row.extend([0.0, 0.0, 0.0])  # Replace with actual per-epoch class metrics if available
        
        writer_csv.writerow(row)

# Save training configuration
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
    'optimization': {
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'mixed_precision': USE_AMP,
        'gradient_clipping': 10.0
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

print(f"\n Exported enhanced training log to: {CSV_PATH.resolve()}")
print(f"  Saved training configuration to: {config_path.resolve()}")
print(f" TensorBoard logs saved to: {LOG_DIR.resolve()}")
print("   Run 'tensorboard --logdir run/logs' to view training progress")

# Close TensorBoard writer
writer.close()

# Enhanced Plotting with additional metrics
plot_metrics(metrics, PLOT_DIR)

# Plot learning rate schedule
plt.figure(figsize=(12, 6))
epochs = list(range(1, len(metrics["learning_rate"]) + 1))
plt.subplot(1, 2, 1)
plt.plot(epochs, metrics["learning_rate"], linewidth=2, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('OneCycleLR Schedule')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Plot training efficiency metrics
plt.subplot(1, 2, 2)
plt.plot(epochs, metrics["val_map"], label='mAP@0.5', linewidth=2)
plt.plot(epochs, metrics["val_precision"], label='Precision', linewidth=2)
plt.plot(epochs, metrics["val_recall"], label='Recall', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "advanced_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Enhanced training complete!")
print(f" Plots saved in: {PLOT_DIR.resolve()}")
print(f" Model weights saved in: {WEIGHT_DIR.resolve()}")
print(f" Training logs saved in: {LOG_DIR.resolve()}")

# Performance summary
print(f"\n{'='*60}")
print(f" FINAL TRAINING RESULTS")
print(f"{'='*60}")
print(f"Best mAP@0.5: {best_map:.4f}")
print(f"Final Training Loss: {metrics['train_loss'][-1]:.4f}")
print(f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}")
print(f"Final Precision: {metrics['val_precision'][-1]:.4f}")
print(f"Final Recall: {metrics['val_recall'][-1]:.4f}")
print(f"Training Features Used:")
print(f"  ✓ Backbone Transfer Learning from {PRETRAINED_WEIGHTS}")
print(f"  ✓ Custom Detection Head (trained from scratch)")
print(f"  ✓ AdamW Optimizer (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"  ✓ OneCycleLR Scheduler")
print(f"  ✓ Mixed Precision Training ({'Enabled' if USE_AMP else 'Disabled'})")
print(f"  ✓ Gradient Accumulation ({ACCUMULATE_STEPS} steps)")
print(f"  ✓ Backbone Freezing ({FREEZE_BACKBONE_EPOCHS} epochs)")
print(f"{'='*60}")

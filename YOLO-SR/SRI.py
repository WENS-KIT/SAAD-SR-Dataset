import os
import csv
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import argparse
from torchmetrics.detection import MeanAveragePrecision
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the multi-scale evaluation."""
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.dataset_root = Path("./scale_dataset")
        self.models_dir = Path("./SAAD_models")
        self.output_dir = Path("./results")
        self.scales = ["15m", "25m", "45m"]  # Default altitude/scale levels
        
        # Model parameters
        self.image_size = 640
        self.batch_size = 4
        self.num_classes = 10
        self.conf_threshold = 0.25
        self.iou_threshold = 0.5
        
        # Evaluation parameters
        self.epsilon = 1e-6  # For SRI calculation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        for key, value in config_data.items():
            if hasattr(self, key):
                if key in ['dataset_root', 'models_dir', 'output_dir']:
                    setattr(self, key, Path(value))
                else:
                    setattr(self, key, value)
    
    def save_config(self, output_path: str):
        """Save current configuration."""
        config_dict = {
            'dataset_root': str(self.dataset_root),
            'models_dir': str(self.models_dir),
            'output_dir': str(self.output_dir),
            'scales': self.scales,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'num_classes': self.num_classes,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'epsilon': self.epsilon
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

class MultiScaleDataset(Dataset):
    """Dataset class for multi-scale evaluation with corrected path resolution."""
    
    def __init__(self, data_root: Path, scale: str, split: str = "val", 
                 img_size: int = 640, transform=None):
        self.data_root = Path(data_root)
        self.scale = scale
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Corrected directory structure based on your dataset YAML files
        self.images_dir = self.data_root / scale / split / "images"
        self.labels_dir = self.data_root / scale / split / "labels"

        self.image_paths = []
        self.label_paths = []

        if self.images_dir.exists():
            for img_path in sorted(self.images_dir.glob("*.jpg")):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)
        else:
            logger.warning(f"Image directory not found: {self.images_dir}")

        logger.info(f"Loaded {len(self.image_paths)} samples for scale {scale} from {self.images_dir}")

        # Define default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Load corresponding label
        label_path = self.label_paths[idx]
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([class_id, x_center, y_center, width, height])
        else:
            logger.warning(f"Missing label file: {label_path}")

        # Apply image transform
        if self.transform:
            image = self.transform(image)

        return image, labels, img_path.name

def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, labels, filenames = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(labels), list(filenames)

class ModelEvaluator:
    """Universal model evaluator for different model types."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize mAP metric
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            iou_thresholds=[0.5],
            class_metrics=True
        )
    
    def load_model(self, model_path: Path) -> Tuple[object, str]:
        """Load model from path and determine its type."""
        model_path = Path(model_path)
        model_name = model_path.name.lower()
        
        try:
            if 'yolo' in model_name or model_path.suffix == '.pt':
                # Try loading as YOLO model
                from ultralytics import YOLO
                model = YOLO(model_path)
                model_type = "yolo"
                logger.info(f"Loaded YOLO model: {model_path.name}")
                
            elif 'torch' in model_name or 'pth' in model_name:
                # Try loading as PyTorch model
                model = torch.load(model_path, map_location=self.device)
                model_type = "pytorch"
                logger.info(f"Loaded PyTorch model: {model_path.name}")
                
            else:
                # Generic loading
                model = torch.load(model_path, map_location=self.device)
                model_type = "generic"
                logger.info(f"Loaded generic model: {model_path.name}")
            
            return model, model_type
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise
    
    def predict_yolo(self, model, images: torch.Tensor) -> List[Dict]:
        """Get predictions from YOLO model."""
        predictions = []
        
        # Convert tensor to list of PIL images or numpy arrays
        images_list = []
        for img in images:
            # Denormalize and convert to numpy
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            images_list.append(img_np)
        
        # Get predictions
        results = model(images_list, verbose=False, conf=self.config.conf_threshold)
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                labels = result.boxes.cls.cpu().long()
                
                pred_dict = {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
            else:
                pred_dict = {
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0,)),
                    'labels': torch.empty((0,), dtype=torch.long)
                }
            
            predictions.append(pred_dict)
        
        return predictions
    
    def predict_pytorch(self, model, images: torch.Tensor) -> List[Dict]:
        """Get predictions from PyTorch model."""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            outputs = model(images.to(self.device))
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # Handle Detectron2-style outputs
                for i in range(len(images)):
                    if 'instances' in outputs:
                        instances = outputs['instances'][i]
                        pred_dict = {
                            'boxes': instances.pred_boxes.tensor.cpu(),
                            'scores': instances.scores.cpu(),
                            'labels': instances.pred_classes.cpu()
                        }
                    else:
                        pred_dict = {
                            'boxes': torch.empty((0, 4)),
                            'scores': torch.empty((0,)),
                            'labels': torch.empty((0,), dtype=torch.long)
                        }
                    predictions.append(pred_dict)
            
            elif isinstance(outputs, (list, tuple)):
                # Handle list/tuple outputs
                for i in range(len(images)):
                    if len(outputs) > i and outputs[i] is not None:
                        output = outputs[i]
                        if len(output) >= 6:  # Assuming [x1, y1, x2, y2, conf, class]
                            boxes = output[:, :4].cpu()
                            scores = output[:, 4].cpu()
                            labels = output[:, 5].cpu().long()
                            
                            pred_dict = {
                                'boxes': boxes,
                                'scores': scores,
                                'labels': labels
                            }
                        else:
                            pred_dict = {
                                'boxes': torch.empty((0, 4)),
                                'scores': torch.empty((0,)),
                                'labels': torch.empty((0,), dtype=torch.long)
                            }
                    else:
                        pred_dict = {
                            'boxes': torch.empty((0, 4)),
                            'scores': torch.empty((0,)),
                            'labels': torch.empty((0,), dtype=torch.long)
                        }
                    predictions.append(pred_dict)
            
            else:
                # Handle tensor outputs
                for i in range(len(images)):
                    pred_dict = {
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty((0,)),
                        'labels': torch.empty((0,), dtype=torch.long)
                    }
                    predictions.append(pred_dict)
        
        return predictions
    
    def evaluate_model_on_scale(self, model, model_type: str, scale: str) -> float:
        """Evaluate model on specific scale."""
        try:
            # Create dataset
            dataset = MultiScaleDataset(
                data_root=self.config.dataset_root,
                scale=scale,
                split="val",
                img_size=self.config.image_size
            )
            
            if len(dataset) == 0:
                logger.warning(f"No data found for scale {scale}")
                return 0.0
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=2
            )
            
            all_predictions = []
            all_targets = []
            
            # Evaluate batches
            for images, labels_batch, filenames in tqdm(dataloader, desc=f"Evaluating {scale}"):
                # Get predictions
                if model_type == "yolo":
                    batch_predictions = self.predict_yolo(model, images)
                else:
                    batch_predictions = self.predict_pytorch(model, images)
                
                # Process targets
                batch_targets = []
                for labels in labels_batch:
                    if labels and len(labels) > 0:
                        labels_tensor = torch.tensor(labels, dtype=torch.float32)
                        
                        # Convert from normalized xywh to xyxy
                        boxes = labels_tensor[:, 1:5].clone()
                        # Convert center coordinates to corner coordinates
                        boxes[:, 0] = (labels_tensor[:, 1] - labels_tensor[:, 3] / 2) * self.config.image_size  # x1
                        boxes[:, 1] = (labels_tensor[:, 2] - labels_tensor[:, 4] / 2) * self.config.image_size  # y1
                        boxes[:, 2] = (labels_tensor[:, 1] + labels_tensor[:, 3] / 2) * self.config.image_size  # x2
                        boxes[:, 3] = (labels_tensor[:, 2] + labels_tensor[:, 4] / 2) * self.config.image_size  # y2
                        
                        labels_cls = labels_tensor[:, 0].long()
                        
                        target_dict = {
                            'boxes': boxes,
                            'labels': labels_cls
                        }
                    else:
                        target_dict = {
                            'boxes': torch.empty((0, 4)),
                            'labels': torch.empty((0,), dtype=torch.long)
                        }
                    
                    batch_targets.append(target_dict)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
            
            # Calculate mAP
            try:
                self.map_metric.reset()
                self.map_metric.update(all_predictions, all_targets)
                map_result = self.map_metric.compute()
                mAP = float(map_result['map_50'].item())
            except Exception as e:
                logger.error(f"Error computing mAP for scale {scale}: {str(e)}")
                mAP = 0.0
            
            logger.info(f"Scale {scale}: mAP@0.5 = {mAP:.4f}")
            return mAP
            
        except Exception as e:
            logger.error(f"Error evaluating scale {scale}: {str(e)}")
            return 0.0

def compute_sri(mAPs: List[float], epsilon: float = 1e-6) -> Tuple[float, float, float]:
    """Compute Scale Robustness Index."""
    if not mAPs or len(mAPs) == 0:
        return 0.0, 0.0, 0.0
    
    mAPs_array = np.array(mAPs)
    mu = float(np.mean(mAPs_array))
    sigma = float(np.std(mAPs_array))
    
    # SRI = 1 - (coefficient of variation)
    sri = 1.0 - (sigma / (mu + epsilon))
    sri = max(0.0, min(1.0, sri))  # Bound between 0 and 1
    
    return mu, sigma, sri

def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. SRI Comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['Model'], results_df['SRI'], alpha=0.8, edgecolor='black')
    plt.title('Scale Robustness Index (SRI) Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('SRI Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sri in zip(bars, results_df['SRI']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sri:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sri_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. mAP vs Scale
    plt.figure(figsize=(12, 8))
    scale_cols = [col for col in results_df.columns if col.startswith('mAP_')]
    scales = [col.replace('mAP_', '') for col in scale_cols]
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        mAPs = [row[col] for col in scale_cols]
        plt.plot(scales, mAPs, marker='o', linewidth=2.5, markersize=8, 
                label=row['Model'], alpha=0.8)
    
    plt.title('Model Performance Across Scales', fontsize=16, fontweight='bold')
    plt.xlabel('Scale/Altitude', fontsize=12)
    plt.ylabel('mAP@0.5', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'map_vs_scale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mean mAP with Standard Deviation
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Model'], results_df['Mean_mAP'], 
            yerr=results_df['Std_mAP'], capsize=8, alpha=0.8, edgecolor='black')
    plt.title('Mean mAP ± Standard Deviation', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean mAP', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_map_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of mAP across scales
    scale_data = results_df[scale_cols].T
    scale_data.columns = results_df['Model']
    scale_data.index = scales
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(scale_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'mAP@0.5'})
    plt.title('mAP Heatmap Across Scales and Models', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Scale/Altitude', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'map_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"All visualizations saved to {output_dir}")

def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Multi-Scale mAP and SRI Calculator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset root directory')
    parser.add_argument('--models', type=str, help='Path to models directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--scales', nargs='+', help='List of scales to evaluate')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config.dataset_root = Path(args.dataset)
    if args.models:
        config.models_dir = Path(args.models)
    if args.output:
        config.output_dir = Path(args.output)
    if args.scales:
        config.scales = args.scales
    
    logger.info("Starting Multi-Scale mAP and SRI Evaluation")
    logger.info(f"Dataset: {config.dataset_root}")
    logger.info(f"Models: {config.models_dir}")
    logger.info(f"Scales: {config.scales}")
    logger.info(f"Device: {config.device}")
    
    # Validate paths
    if not config.dataset_root.exists():
        logger.error(f"Dataset directory not found: {config.dataset_root}")
        return
    
    if not config.models_dir.exists():
        logger.error(f"Models directory not found: {config.models_dir}")
        return
    
    # Find model files
    model_files = []
    for ext in ['*.pt', '*.pth', '*.onnx', '*.pkl']:
        model_files.extend(list(config.models_dir.glob(ext)))
    
    if not model_files:
        logger.error(f"No model files found in {config.models_dir}")
        return
    
    logger.info(f"Found {len(model_files)} models to evaluate")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Results storage
    results = []
    
    # CSV file setup
    csv_path = config.output_dir / 'multiscale_results.csv'
    csv_columns = ['Model', 'Mean_mAP', 'Std_mAP', 'SRI'] + [f'mAP_{scale}' for scale in config.scales]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
    
    # Evaluate each model
    for model_path in sorted(model_files):
        logger.info(f"\nEvaluating: {model_path.name}")
        
        try:
            # Load model
            model, model_type = evaluator.load_model(model_path)
            
            # Evaluate on each scale
            scale_mAPs = []
            for scale in config.scales:
                mAP = evaluator.evaluate_model_on_scale(model, model_type, scale)
                scale_mAPs.append(mAP)
            
            # Compute statistics
            mean_mAP, std_mAP, sri = compute_sri(scale_mAPs, config.epsilon)
            
            # Store results
            result = {
                'model': model_path.name,
                'mean_mAP': mean_mAP,
                'std_mAP': std_mAP,
                'SRI': sri,
                'scale_mAPs': scale_mAPs,
                'model_type': model_type
            }
            results.append(result)
            
            # Write to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [model_path.name, f"{mean_mAP:.4f}", f"{std_mAP:.4f}", f"{sri:.4f}"]
                row.extend([f"{mAP:.4f}" for mAP in scale_mAPs])
                writer.writerow(row)
            
            logger.info(f"Results - Mean mAP: {mean_mAP:.4f}, Std: {std_mAP:.4f}, SRI: {sri:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_path.name}: {str(e)}")
            continue
    
    # Generate summary
    if results:
        print("\n" + "=" * 100)
        print(f"{'Model':<30} {'Mean mAP':<12} {'Std mAP':<12} {'SRI':<12} {'Type':<15}")
        print("-" * 100)
        
        for result in results:
            print(f"{result['model']:<30} {result['mean_mAP']:<12.4f} "
                  f"{result['std_mAP']:<12.4f} {result['SRI']:<12.4f} {result['model_type']:<15}")
        
        print("=" * 100)
        
        # Find best models
        best_sri = max(results, key=lambda x: x['SRI'])
        best_map = max(results, key=lambda x: x['mean_mAP'])
        most_stable = min(results, key=lambda x: x['std_mAP'])
        
        print(f"\n🏆 Best SRI: {best_sri['model']} (SRI: {best_sri['SRI']:.4f})")
        print(f"🎯 Best Mean mAP: {best_map['model']} (mAP: {best_map['mean_mAP']:.4f})")
        print(f"📊 Most Stable: {most_stable['model']} (Std: {most_stable['std_mAP']:.4f})")
        
        # Create visualizations
        results_df = pd.read_csv(csv_path)
        create_visualizations(results_df, config.output_dir)
        
        # Export detailed results
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'dataset_root': str(config.dataset_root),
                'scales': config.scales,
                'image_size': config.image_size,
                'batch_size': config.batch_size,
                'num_classes': config.num_classes,
                'conf_threshold': config.conf_threshold,
                'iou_threshold': config.iou_threshold
            },
            'summary': {
                'total_models': len(results),
                'best_sri_model': best_sri['model'],
                'best_sri_score': best_sri['SRI'],
                'best_map_model': best_map['model'],
                'best_map_score': best_map['mean_mAP']
            },
            'detailed_results': results
        }
        
        with open(config.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save configuration used
        config.save_config(config.output_dir / 'config_used.yaml')
        
        logger.info(f"\nEvaluation completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info(f"CSV file: {csv_path}")
    
    else:
        logger.error("No models were successfully evaluated!")

if __name__ == "__main__":
    main()

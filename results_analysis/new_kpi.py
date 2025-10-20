import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR


def evaluate_model(model_type, weights_path, data_yaml_path):
    # Load model
    if model_type.lower() == 'yolo':
        model = YOLO(weights_path)
    elif model_type.lower() == 'rtdetr':
        model = RTDETR(weights_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'yolo' or 'rtdetr'.")

    # Run evaluation at IoU=0.5
    metrics = model.val(
        data=data_yaml_path,
        iou=0.5,
        split='val',
        save=False,
        save_json=False,
        plots=False,
        rect=True,
        verbose=True
    )

    # Access metrics (all are floats, NOT functions!)
    map50 = metrics.box.map50
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Inference speed
    inference_time_ms = metrics.speed['inference']
    inference_fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

    # Print summary
    print("\n" + "=" * 50)
    print(f"✅ [{model_type.upper()}] Evaluation Summary")
    print("=" * 50)
    print(f"mAP@0.5           : {map50:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1_score:.4f}")
    print(f"Inference Time    : {inference_time_ms:.2f} ms/image")
    print(f"Approx FPS        : {inference_fps:.2f} frames/sec")
    print("=" * 50 + "\n")

    return map50, precision, recall, f1_score, inference_time_ms, inference_fps


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO or RT-DETR model (mAP@0.5, Precision, Recall, F1, Inference Speed)")
    parser.add_argument('--model', type=str, choices=['yolo', 'rtdetr'], required=True,
                        help="Model type: 'yolo' or 'rtdetr'")
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weight file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    args = parser.parse_args()

    # Validate paths
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weight file not found: {args.weights}")
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"YAML file not found: {args.data}")

    evaluate_model(args.model, args.weights, args.data)


if __name__ == '__main__':
    main()


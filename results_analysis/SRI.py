import numpy as np

def compute_sri(maps, epsilon=1e-6):
    """
    Computes mean mAP (μ), standard deviation (σ), and Scale Robustness Index (SRI)
    from a list of mAP@0.5 values.

    Args:
        maps (list): List of (altitude_label, mAP@0.5) tuples.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        tuple: (mean mAP, standard deviation, SRI)
    """
    values = [val for _, val in maps]
    values = np.array(values)
    mu = np.mean(values)
    sigma = np.std(values, ddof=0)
    sri = 1 - (sigma / (mu + epsilon))
    return round(mu, 4), round(sigma, 4), round(sri, 4)

def print_sri_report(model_name, maps):
    print("=" * 50)
    for altitude, value in maps:
        print(f"[{model_name}] Evaluation Summary ({altitude})")
        print("=" * 50)
        print(f"mAP@0.5           : {value:.4f}\n")

    mu, sigma, sri = compute_sri(maps)
    print("=" * 50)
    print(f"[{model_name}] Scale Robustness Summary")
    print("=" * 50)
    print(f"Mean mAP (μ)      : {mu:.4f}")
    print(f"Std Dev (σ)       : {sigma:.4f}")
    print(f"SRI               : {sri:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # Example input for YOLO11
    yolo11_maps = [
        ("15m", 0.8572),
        ("25m", 0.9024),
        ("45m", 0.8922)
    ]

    print_sri_report("RT-DETR", yolo11_maps)


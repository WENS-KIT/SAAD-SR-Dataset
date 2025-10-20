import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Path to training CSVs
CSV_DIR = "/home/imad/Documents/altitude_training"

# Map CSV filenames to clean model labels
model_name_map = {
    'rtdetr.csv': 'RT-DETR',
    'v8.csv': 'YOLOv8',
    'yolov10.csv': 'YOLOv10',
    'yolov11.csv': 'YOLOv11',
    'yolov8-sr.csv': 'YOLO-SR'
}

# Metrics to visualize
metrics_to_plot = [
    'train/giou_loss', 'train/cls_loss', 'train/l1_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/giou_loss', 'val/cls_loss', 'val/l1_loss'
]

# Marker and style settings
marker_styles = ['o', 's', '^', 'D', 'v']
sns.set_style("whitegrid")
colors = sns.color_palette("Set1", n_colors=len(model_name_map))

# OCDet-style font and layout
rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14
})

# Create output directories
os.makedirs("plots/metrics", exist_ok=True)
os.makedirs("plots/tables", exist_ok=True)

def save_table_as_image(df, filename, title="Comparison Table"):
    fig, ax = plt.subplots(figsize=(7, 1 + 0.4 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax.set_title(title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

def plot_metrics(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    dataframes = {}
    model_markers = {}
    model_colors = {}

    # Load each model's data
    for i, file in enumerate(files):
        filepath = os.path.join(folder, file)
        df = pd.read_csv(filepath)
        model_name = model_name_map.get(file, file.replace('.csv', ''))
        dataframes[model_name] = df
        model_markers[model_name] = marker_styles[i % len(marker_styles)]
        model_colors[model_name] = colors[i % len(colors)]

    # Plot all metrics
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 9))
        ax = plt.gca()
        summary_rows = []

        for model_name, df in dataframes.items():
            if metric not in df.columns:
                continue

            x = df['epoch']
            y = df[metric]

            plt.plot(
                x, y,
                label=model_name,
                linestyle='-',  # Solid lines only
                linewidth=2.5,
                marker=model_markers[model_name],
                markersize=7,
                markevery=10,
                alpha=0.95,
                color=model_colors[model_name]
            )

            summary_rows.append({
                "Model": model_name,
                "Final Epoch": x.iloc[-1],
                "Final Value": round(y.iloc[-1], 6)
            })

        # Axis labels and title
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel(metric.split("/")[-1].replace("_", " ").capitalize(), fontsize=16)
        plt.title(metric.replace("/", " ").capitalize(), fontsize=18, weight="bold")

        # Denser, clearer grid
        ax.minorticks_on()
        ax.grid(which='major', linestyle=':', linewidth=1.0, alpha=0.85)
        ax.grid(which='minor', linestyle=':', linewidth=0.6, alpha=0.65)

        # Legend position: top right for loss, bottom right otherwise
        if "loss" in metric.lower():
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")

        # Save figure
        plot_path = f"plots/metrics/{metric.replace('/', '_')}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=600)
        plt.close()

        # Save table summary
        summary_df = pd.DataFrame(summary_rows)
        table_path = f"plots/tables/{metric.replace('/', '_')}_table.png"
        save_table_as_image(summary_df, table_path, title=f"{metric} Summary")

    print("✅ All OCDet-style plots saved with updated legend positions.")

# Main
if __name__ == "__main__":
    plot_metrics(CSV_DIR)


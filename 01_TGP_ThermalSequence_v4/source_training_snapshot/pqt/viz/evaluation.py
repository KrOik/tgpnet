from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_results(
    targets: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
    scatter_path: Path,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    scatter_path = Path(scatter_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(16, 10))
    ax1 = figure.add_subplot(2, 2, 1)
    ax1.scatter(targets, predictions, alpha=0.6)
    ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", linewidth=2)
    ax1.set_title(f"Predicted vs Actual (RMSE={metrics['rmse']:.4f})")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")

    residuals = predictions - targets
    ax2 = figure.add_subplot(2, 2, 2)
    ax2.scatter(predictions, residuals, alpha=0.6)
    ax2.axhline(y=0.0, color="r", linestyle="--")
    ax2.set_title(f"Residuals (Bias={metrics['bias']:.4f})")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")

    ax3 = figure.add_subplot(2, 2, 3)
    ax3.hist(np.abs(residuals), bins=30)
    ax3.set_title(f"Absolute Error (MAE={metrics['mae']:.4f})")
    ax3.set_xlabel("Absolute Error")

    ax4 = figure.add_subplot(2, 2, 4)
    ax4.hist(residuals, bins=30)
    ax4.set_title("Residual Distribution")
    ax4.set_xlabel("Residual")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    scatter = plt.figure(figsize=(8, 8))
    axis = scatter.add_subplot(1, 1, 1)
    axis.scatter(targets, predictions, alpha=0.6)
    axis.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", linewidth=2)
    axis.set_title("Prediction Scatter")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    scatter.tight_layout()
    scatter.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(scatter)
    return output_path, scatter_path


def write_metrics_summary(metrics: dict[str, float], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Model Evaluation Metrics Summary",
        "=" * 50,
        f"Sample Size: {int(metrics['sample_size'])}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"MAE: {metrics['mae']:.4f}",
        f"MAPE: {metrics['mape']:.2f}%",
        f"R2: {metrics['r2']:.4f}",
        f"Bias: {metrics['bias']:.4f}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(24, 5))
    total_axes = 4

    ax1 = figure.add_subplot(1, total_axes, 1)
    ax1.plot(history.get("train_loss", []), label="Train Loss")
    ax1.plot(history.get("val_loss", []), label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2 = figure.add_subplot(1, total_axes, 2)
    ax2.plot(history.get("learning_rates", []))
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.grid(True)

    ax3 = figure.add_subplot(1, total_axes, 3)
    ax3.plot(history.get("prediction_bias", []))
    ax3.axhline(y=0.0, color="r", linestyle="--")
    ax3.set_title("Prediction Bias")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Bias")
    ax3.grid(True)

    ax4 = figure.add_subplot(1, total_axes, 4)
    ax4.plot(history.get("val_mae", []), label="Validation MAE", color="#1b9e77")
    ax4.set_title("Spatial Validation MAE")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MAE")
    ax4.legend()
    ax4.grid(True)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path

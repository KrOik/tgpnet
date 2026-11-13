from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction_histogram(predictions: pd.Series, dataset_name: str, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.hist(predictions, bins=min(40, max(10, len(predictions) // 25)), color="#2878b5", alpha=0.85)
    axis.set_title(f"{dataset_name} Prediction Distribution")
    axis.set_xlabel("Predicted Value")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path

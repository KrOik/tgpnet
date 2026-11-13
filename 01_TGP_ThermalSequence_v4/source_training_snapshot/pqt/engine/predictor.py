from __future__ import annotations

import pandas as pd
import torch
from copy import deepcopy

from pqt.config.schema import DataConfig, PredictionConfig
from pqt.data.loaders import create_image_prediction_loader
from pqt.model.checkpoint import load_checkpoint
from pqt.model.factory import build_model
from pqt.model.super_resolution import FrozenEDSRx2
from pqt.experiments.contracts import ExperimentRuntime
from pqt.experiments.loader import apply_experiment_model_config, resolve_experiment

from .runtime import resolve_device


def predict_directory(
    config: PredictionConfig,
    experiment: ExperimentRuntime | None = None,
) -> dict[str, object]:
    config = deepcopy(config)
    if config.checkpoint_path is None:
        raise ValueError("Prediction requires checkpoint_path.")
    if config.image_dir is None:
        raise ValueError("Prediction requires image_dir.")

    experiment = experiment or resolve_experiment(config.experiment)
    config.model = apply_experiment_model_config(config.model, experiment)
    device = resolve_device(config.runtime)
    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    data_config = DataConfig(
        image_dir=config.image_dir,
        input_mode=config.input_mode,
        image_size=(1024, 1024),
    )
    loader = create_image_prediction_loader(data_config, config.runtime)
    model = build_model(config.model, experiment=experiment).to(device)
    if config.model.sr_weights_path is None:
        raise ValueError("Prediction requires sr_weights_path.")
    sr = FrozenEDSRx2(config.model.sr_weights_path).to(device).eval()
    metadata = load_checkpoint(config.checkpoint_path, model=model, map_location=device)
    model.eval()

    filenames: list[str] = []
    predictions: list[float] = []
    amp_enabled = config.runtime.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for inputs, batch_filenames in loader:
            original = inputs.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(sr(original)).cpu().flatten().tolist()
            filenames.extend(batch_filenames)
            predictions.extend(outputs)

    results = pd.DataFrame({"Image_Name": filenames, "Predicted_Value": predictions})
    csv_path = config.output.output_dir / config.output.prediction_csv_name
    results.to_csv(csv_path, index=False)
    return {"metadata": metadata, "results_path": csv_path, "results": results}

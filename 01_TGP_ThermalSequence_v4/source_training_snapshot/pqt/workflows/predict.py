from __future__ import annotations

from pqt.config.defaults import build_default_prediction_config
from pqt.config.schema import PredictionConfig
from pqt.engine.predictor import predict_directory
from pqt.experiments.contracts import ExperimentRuntime


def run_prediction(
    config: PredictionConfig | None = None,
    experiment: ExperimentRuntime | None = None,
) -> dict[str, object]:
    active_config = config or build_default_prediction_config()
    return predict_directory(active_config, experiment=experiment)

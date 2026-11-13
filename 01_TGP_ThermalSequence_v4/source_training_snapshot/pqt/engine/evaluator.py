from __future__ import annotations

from pqt.config.schema import EvaluationConfig
from pqt.experiments.contracts import ExperimentRuntime
from .standard_evaluation import run_standard_evaluation


def evaluate_model(
    config: EvaluationConfig,
    experiment: ExperimentRuntime | None = None,
) -> dict[str, object]:
    return run_standard_evaluation(config, experiment=experiment)

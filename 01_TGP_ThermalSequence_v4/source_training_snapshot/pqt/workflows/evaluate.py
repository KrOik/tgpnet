from __future__ import annotations

from pqt.config.defaults import build_default_standard_evaluation_config
from pqt.config.schema import EvaluationConfig
from pqt.engine.evaluator import evaluate_model
from pqt.experiments.contracts import ExperimentRuntime


def run_evaluation(
    config: EvaluationConfig | None = None,
    experiment: ExperimentRuntime | None = None,
) -> dict[str, object]:
    active_config = config or build_default_standard_evaluation_config()
    return evaluate_model(active_config, experiment=experiment)

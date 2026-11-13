from __future__ import annotations

from copy import deepcopy

from pqt.config.defaults import build_default_train_config
from pqt.config.schema import TrainConfig
from pqt.engine.trainer import train_model
from pqt.experiments.contracts import ExperimentRuntime


def run_training(
    config: TrainConfig | None = None,
    experiment_id: str | None = None,
    experiment: ExperimentRuntime | None = None,
    *,
    config_is_resolved: bool = False,
) -> dict[str, object]:
    active_config = deepcopy(config) if config is not None else build_default_train_config()
    if experiment_id is not None:
        active_config.experiment.experiment_id = experiment_id
    return train_model(
        active_config,
        experiment=experiment,
        apply_experiment_config=not config_is_resolved,
    )

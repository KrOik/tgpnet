from .config.defaults import (
    build_default_evaluation_config,
    build_default_standard_evaluation_config,
    build_default_prediction_config,
    build_default_train_config,
)
from .config.schema import (
    BestCheckpointConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    PredictionConfig,
    RuntimeConfig,
    SplitConfig,
    StandardEvaluationDatasetSpec,
    TrainConfig,
)


def build_model(*args, **kwargs):
    from .model.factory import build_model as _build_model

    return _build_model(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    from .workflows.evaluate import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


def run_prediction(*args, **kwargs):
    from .workflows.predict import run_prediction as _run_prediction

    return _run_prediction(*args, **kwargs)


def run_sim_dataset_packaging(*args, **kwargs):
    from .workflows.package_sim_dataset import run_sim_dataset_packaging as _run_sim_dataset_packaging

    return _run_sim_dataset_packaging(*args, **kwargs)


def run_training(*args, **kwargs):
    from .workflows.train import run_training as _run_training

    return _run_training(*args, **kwargs)

__all__ = [
    "BestCheckpointConfig",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OptimizationConfig",
    "OutputConfig",
    "PredictionConfig",
    "RuntimeConfig",
    "SplitConfig",
    "StandardEvaluationDatasetSpec",
    "TrainConfig",
    "build_default_evaluation_config",
    "build_default_standard_evaluation_config",
    "build_default_prediction_config",
    "build_default_train_config",
    "build_model",
    "run_evaluation",
    "run_prediction",
    "run_sim_dataset_packaging",
    "run_training",
]

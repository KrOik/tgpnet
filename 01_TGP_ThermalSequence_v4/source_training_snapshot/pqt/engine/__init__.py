from .metrics import calculate_regression_metrics


def evaluate_model(*args, **kwargs):
    from .evaluator import evaluate_model as _evaluate_model

    return _evaluate_model(*args, **kwargs)


def predict_directory(*args, **kwargs):
    from .predictor import predict_directory as _predict_directory

    return _predict_directory(*args, **kwargs)


def train_model(*args, **kwargs):
    from .trainer import train_model as _train_model

    return _train_model(*args, **kwargs)

__all__ = [
    "calculate_regression_metrics",
    "evaluate_model",
    "predict_directory",
    "train_model",
]

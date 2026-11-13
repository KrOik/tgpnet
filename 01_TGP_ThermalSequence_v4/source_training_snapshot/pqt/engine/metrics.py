from __future__ import annotations

import math

import torch


def calculate_ordered_depth_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_origin: float = 0.24,
    class_step: float = 0.16,
) -> dict[str, float]:
    predictions = predictions.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    predicted_classes = torch.round((predictions - class_origin) / class_step)
    target_classes = torch.round((targets - class_origin) / class_step)
    class_distance = torch.abs(predicted_classes - target_classes)
    return {
        "depth_class_accuracy": (class_distance == 0).float().mean().item(),
        "depth_within_one_class": (class_distance <= 1).float().mean().item(),
        "depth_mean_class_distance": class_distance.float().mean().item(),
    }


def calculate_offgrid_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_origin: float = 0.24,
    class_step: float = 0.16,
    off_grid_tolerance: float = 0.04,
) -> dict[str, float]:
    """Quantify how much predictions escape the discrete training label grid.

    ``off_grid_rate`` is the fraction of predictions more than
    ``off_grid_tolerance`` away from the nearest class level; a perfectly
    grid-snapping model scores 0. ``within_level_std`` averages the prediction
    std inside each true-label class, exposing intra-class resolution.
    """
    predictions = predictions.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    nearest_level = class_origin + torch.round((predictions - class_origin) / class_step) * class_step
    off_grid = torch.abs(predictions - nearest_level) > off_grid_tolerance
    per_level_std: list[torch.Tensor] = []
    for level in torch.unique(targets):
        mask = targets == level
        if int(mask.sum()) > 1:
            per_level_std.append(predictions[mask].std(unbiased=False))
    within_std = torch.stack(per_level_std).mean().item() if per_level_std else 0.0
    return {
        "off_grid_rate": off_grid.float().mean().item(),
        "within_level_std": within_std,
    }


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    predictions = predictions.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    errors = predictions - targets
    mae = torch.abs(errors).mean().item()
    rmse = math.sqrt(torch.square(errors).mean().item())
    bias = errors.mean().item()

    non_zero_mask = targets != 0
    if torch.any(non_zero_mask):
        mape = (
            torch.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])
            .mean()
            .item()
            * 100.0
        )
    else:
        mape = 0.0

    target_mean = targets.mean()
    total_sum_squares = torch.square(targets - target_mean).sum().item()
    residual_sum_squares = torch.square(errors).sum().item()
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "r2": r2,
        "sample_size": float(targets.numel()),
    }

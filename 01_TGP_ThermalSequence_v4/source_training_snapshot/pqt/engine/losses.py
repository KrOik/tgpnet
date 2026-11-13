from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from pqt.model.outputs import RegressionOutput, as_regression_output


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    raw_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    state: dict[str, torch.Tensor] = field(default_factory=dict)


def compute_loss_breakdown(
    criterion: nn.Module,
    predictions: torch.Tensor | RegressionOutput,
    targets: torch.Tensor,
) -> LossBreakdown:
    detailed = getattr(criterion, "forward_with_breakdown", None)
    if callable(detailed):
        result = detailed(predictions, targets)
        if not isinstance(result, LossBreakdown):
            raise TypeError("forward_with_breakdown() must return LossBreakdown.")
        return result
    total = criterion(predictions, targets)
    return LossBreakdown(total=total, raw_terms={"total": total}, weighted_terms={"total": total}, terms={"total": total})


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 0.0) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions: torch.Tensor | RegressionOutput, targets: torch.Tensor) -> torch.Tensor:
        return self.forward_with_breakdown(predictions, targets).total

    def forward_with_breakdown(
        self, predictions: torch.Tensor | RegressionOutput, targets: torch.Tensor
    ) -> LossBreakdown:
        predictions = as_regression_output(predictions).prediction
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        weighted_mse = self.mse_weight * mse
        weighted_l1 = self.l1_weight * l1
        total = weighted_mse + weighted_l1
        return LossBreakdown(
            total=total,
            raw_terms={"mse": mse, "l1": l1},
            weighted_terms={"mse": weighted_mse, "l1": weighted_l1},
            terms={"mse": mse, "l1": l1, "weighted_mse": weighted_mse, "weighted_l1": weighted_l1, "total": total},
        )


class HuberRegressionLoss(nn.Module):
    """Single robust regression objective used by the active v1 path."""

    def __init__(self, delta: float = 0.08) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, predictions: torch.Tensor | RegressionOutput, targets: torch.Tensor) -> torch.Tensor:
        return self.forward_with_breakdown(predictions, targets).total

    def forward_with_breakdown(
        self, predictions: torch.Tensor | RegressionOutput, targets: torch.Tensor
    ) -> LossBreakdown:
        prediction = as_regression_output(predictions).prediction
        value = nn.functional.huber_loss(prediction, targets, reduction="mean", delta=self.delta)
        return LossBreakdown(
            total=value,
            raw_terms={"huber": value},
            weighted_terms={"huber": value},
            terms={"huber": value, "total": value},
        )

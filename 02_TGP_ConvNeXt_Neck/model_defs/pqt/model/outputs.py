from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class RegressionOutput:
    """Explicit model output used by training-aware heads and losses."""

    prediction: torch.Tensor
    auxiliary: dict[str, Any] = field(default_factory=dict)


def as_regression_output(value: torch.Tensor | RegressionOutput) -> RegressionOutput:
    if isinstance(value, RegressionOutput):
        return value
    return RegressionOutput(prediction=value)

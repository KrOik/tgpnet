from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pqt.config.schema import TrainConfig
from pqt.depth_ordinal import ordinal_targets
from pqt.engine.losses import LossBreakdown
from pqt.model.outputs import as_regression_output


class OrdinalHuberLoss(nn.Module):
    """Huber on the scalar prediction plus auxiliary ordinal BCE on the
    cumulative-threshold logits emitted by OrdinalExpectationHead."""

    def __init__(self, delta: float = 0.08, ordinal_weight: float = 0.05) -> None:
        super().__init__()
        self.delta = float(delta)
        self.ordinal_weight = float(ordinal_weight)

    def forward(self, predictions, targets) -> torch.Tensor:
        return self.forward_with_breakdown(predictions, targets).total

    def forward_with_breakdown(self, predictions, targets) -> LossBreakdown:
        output = as_regression_output(predictions)
        huber = F.huber_loss(output.prediction, targets, reduction="mean", delta=self.delta)
        logits = output.auxiliary.get("ordinal_logits")
        if logits is None:
            raise ValueError("OrdinalHuberLoss requires auxiliary['ordinal_logits'].")
        ordinal = F.binary_cross_entropy_with_logits(logits, ordinal_targets(targets))
        weighted_ordinal = self.ordinal_weight * ordinal
        return LossBreakdown(
            total=huber + weighted_ordinal,
            raw_terms={"huber": huber, "ordinal_bce": ordinal},
            weighted_terms={"huber": huber, "ordinal_bce": weighted_ordinal},
            terms={"huber": huber, "ordinal_bce": ordinal, "weighted_ordinal_bce": weighted_ordinal, "total": huber + weighted_ordinal},
        )


def build_loss(config: TrainConfig) -> nn.Module:
    return OrdinalHuberLoss(
        delta=config.optimization.huber_delta,
        ordinal_weight=config.optimization.ordinal_loss_weight,
    )

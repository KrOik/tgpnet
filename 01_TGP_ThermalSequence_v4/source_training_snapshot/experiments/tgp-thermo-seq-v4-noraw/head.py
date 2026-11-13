from __future__ import annotations

import torch
from torch import nn

from pqt.config.schema import ModelConfig
from pqt.depth_ordinal import DEPTH_LEVELS, MAX_RESIDUAL, decode_ordinal_logits
from pqt.model.outputs import RegressionOutput


class OrdinalExpectationHead(nn.Module):
    """Ordinal readout: 8 cumulative-threshold logits over the 9 depth levels.

    The scalar prediction is the level expectation plus a bounded residual
    correction (|residual| <= MAX_RESIDUAL, zero-initialized so the head starts
    as a pure ordinal readout). Ordinal logits are exposed via ``auxiliary``
    for the paired ordinal BCE loss.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128) -> None:
        super().__init__()
        num_thresholds = len(DEPTH_LEVELS) - 1
        self.scores = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_thresholds),
        )
        self.residual = nn.Linear(in_features, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> RegressionOutput:
        logits = self.scores(x)
        _, _, coarse = decode_ordinal_logits(logits)
        prediction = coarse + MAX_RESIDUAL * torch.tanh(self.residual(x))
        return RegressionOutput(prediction=prediction, auxiliary={"ordinal_logits": logits})


def build_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    return OrdinalExpectationHead(input_dim)

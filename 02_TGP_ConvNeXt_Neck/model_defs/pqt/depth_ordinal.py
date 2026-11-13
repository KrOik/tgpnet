from __future__ import annotations

from collections.abc import Sequence

import torch


DEPTH_LEVELS = (0.24, 0.40, 0.56, 0.72, 0.88, 1.04, 1.20, 1.36, 1.52)
DEPTH_THRESHOLDS = (0.32, 0.48, 0.64, 0.80, 0.96, 1.12, 1.28, 1.44)
MAX_RESIDUAL = 0.08


def ordinal_targets(
    targets: torch.Tensor,
    thresholds: Sequence[float] = DEPTH_THRESHOLDS,
) -> torch.Tensor:
    threshold_tensor = targets.new_tensor(tuple(thresholds)).view(1, -1)
    return (targets.view(-1, 1) > threshold_tensor).to(targets.dtype)


def decode_ordinal_logits(
    logits: torch.Tensor,
    levels: Sequence[float] = DEPTH_LEVELS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[1] != len(levels) - 1:
        raise ValueError(f"Expected [batch, {len(levels) - 1}] ordinal logits, got {tuple(logits.shape)}.")
    cumulative = torch.sigmoid(logits)
    probabilities = torch.cat(
        (
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        dim=1,
    )
    level_tensor = logits.new_tensor(tuple(levels)).view(1, -1)
    coarse = (probabilities * level_tensor).sum(dim=1, keepdim=True)
    return cumulative, probabilities, coarse

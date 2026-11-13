from __future__ import annotations

from typing import NamedTuple

import torch


class DualResolutionInput(NamedTuple):
    original: torch.Tensor
    super_resolved: torch.Tensor


def move_model_input(value, device: torch.device | str):
    if isinstance(value, DualResolutionInput):
        return DualResolutionInput(value.original.to(device), value.super_resolved.to(device))
    return value.to(device)


def slice_model_input(value, stop: int):
    if isinstance(value, DualResolutionInput):
        return DualResolutionInput(value.original[:stop], value.super_resolved[:stop])
    return value[:stop]


def model_input_batch_size(value) -> int:
    return int(value.original.shape[0] if isinstance(value, DualResolutionInput) else value.shape[0])


def primary_model_input(value) -> torch.Tensor:
    return value.original if isinstance(value, DualResolutionInput) else value

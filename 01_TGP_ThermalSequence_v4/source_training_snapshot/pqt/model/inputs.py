from __future__ import annotations

from typing import NamedTuple

import torch


class DualResolutionInput(NamedTuple):
    original: torch.Tensor
    super_resolved: torch.Tensor


class NeighborhoodInput(NamedTuple):
    """One center stripe image plus context curves from nearby grid points.

    ``neighbor_curves`` is [B, K, L] normalized to [-1, 1]; ``distances`` is
    [B, K] in grid units; ``mask`` is a boolean [B, K] selecting real
    neighbors vs zero padding.
    """

    center: torch.Tensor
    neighbor_curves: torch.Tensor
    distances: torch.Tensor
    mask: torch.Tensor


def move_model_input(value, device: torch.device | str):
    if isinstance(value, DualResolutionInput):
        return DualResolutionInput(value.original.to(device), value.super_resolved.to(device))
    if isinstance(value, NeighborhoodInput):
        return NeighborhoodInput(
            value.center.to(device),
            value.neighbor_curves.to(device),
            value.distances.to(device),
            value.mask.to(device),
        )
    return value.to(device)


def slice_model_input(value, stop: int):
    if isinstance(value, DualResolutionInput):
        return DualResolutionInput(value.original[:stop], value.super_resolved[:stop])
    if isinstance(value, NeighborhoodInput):
        return NeighborhoodInput(
            value.center[:stop],
            value.neighbor_curves[:stop],
            value.distances[:stop],
            value.mask[:stop],
        )
    return value[:stop]


def model_input_batch_size(value) -> int:
    if isinstance(value, DualResolutionInput):
        return int(value.original.shape[0])
    if isinstance(value, NeighborhoodInput):
        return int(value.center.shape[0])
    return int(value.shape[0])


def primary_model_input(value) -> torch.Tensor:
    if isinstance(value, DualResolutionInput):
        return value.original
    if isinstance(value, NeighborhoodInput):
        return value.center
    return value

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from pqt.model.inputs import NeighborhoodInput

from .datasets import load_image


_COORD_RE = re.compile(r"^(?P<X>-?\d+)_(?P<Y>-?\d+)\.[^.]+$")


def parse_grid_coords(names: list[str]) -> np.ndarray:
    """Parse ``<x>_<y>.<ext>`` filenames into an [N, 2] float coordinate array."""
    parts = np.array(
        [match.groups() if (match := _COORD_RE.match(name)) else None for name in names],
        dtype=object,
    )
    if (parts == None).any():  # noqa: E711 - object-array None check
        bad = [name for name, part in zip(names, parts, strict=True) if part is None][:5]
        raise ValueError(f"cannot parse grid coordinates from filenames: {bad}")
    return np.asarray(parts.tolist(), dtype=np.float64)


def select_neighbors(coords: np.ndarray, index: int, count: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of up to ``count`` nearest points within ``radius``."""
    deltas = coords - coords[index]
    distances = np.sqrt((deltas * deltas).sum(axis=1))
    distances[index] = np.inf
    order = np.argsort(distances, kind="stable")
    selected = order[distances[order] <= radius][:count]
    return selected, distances[selected]


class _NeighborCurveMixin:
    """Shared neighbor lookup + curve loading for context datasets."""

    image_dir: Path
    neighbor_count: int
    radius: float
    transform: Callable | None
    input_mode: str
    coords: np.ndarray
    names: list[str]

    def _init_context(self, image_dir: Path, neighbor_count: int, radius: float, transform, input_mode: str) -> None:
        self.image_dir = Path(image_dir)
        self.neighbor_count = int(neighbor_count)
        self.radius = float(radius)
        self.transform = transform
        self.input_mode = input_mode
        self._curve_cache: dict[str, torch.Tensor] = {}

    def _curve(self, name: str) -> torch.Tensor:
        cached = self._curve_cache.get(name)
        if cached is None:
            image = load_image(self.image_dir / name, self.input_mode)
            tensor = self.transform(image) if self.transform is not None else image
            cached = tensor.mean(dim=(0, 1))
            self._curve_cache[name] = cached
        return cached

    def _context(self, index: int):
        selected, distances = select_neighbors(self.coords, index, self.neighbor_count, self.radius)
        if selected.size:
            curves = torch.stack([self._curve(self.names[int(j)]) for j in selected])
            dist = torch.from_numpy(distances.astype(np.float32))
        else:
            width = self._curve(self.names[index]).shape[-1]
            curves = torch.zeros(0, width)
            dist = torch.zeros(0)
        mask = torch.ones(selected.size, dtype=torch.bool)
        pad = self.neighbor_count - selected.size
        if pad > 0:
            curves = torch.cat([curves, torch.zeros(pad, curves.shape[-1])], dim=0)
            dist = torch.cat([dist, torch.full((pad,), float("inf"), dtype=torch.float32)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
        return curves, dist, mask


class NeighborContextDataset(_NeighborCurveMixin, Dataset):
    """Wrap a supervised dataset so each item carries its grid neighbors.

    The neighbor pool is the wrapped dataset's own frame: train items draw
    neighbors only from the train split, val only from val. Neighbor curves are
    normalized [-1, 1] row means matching the image transform width.
    """

    def __init__(
        self,
        base_dataset,
        *,
        image_dir: Path,
        filename_column: str,
        neighbor_count: int,
        radius: float,
        transform,
        input_mode: str,
    ) -> None:
        self.base = base_dataset
        self.names = base_dataset.dataframe[filename_column].astype(str).tolist()
        self.coords = parse_grid_coords(self.names)
        self._init_context(image_dir, neighbor_count, radius, transform, input_mode)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        image, rest = item[0], item[1]
        curves, distances, mask = self._context(index)
        return NeighborhoodInput(image, curves, distances, mask), rest


class FolderNeighborContextDataset(_NeighborCurveMixin, Dataset):
    """Neighbor-context variant of ``ImageFolderDataset`` for prediction runs.

    Requires every filename in the folder to match ``<x>_<y>.<ext>``; callers
    should only enable it when that convention holds (all manifest datasets
    do). The neighbor pool is the folder itself.
    """

    def __init__(self, base_dataset, *, image_dir: Path, neighbor_count: int, radius: float, transform, input_mode: str) -> None:
        self.base = base_dataset
        self.names = list(base_dataset.image_files)
        self.coords = parse_grid_coords(self.names)
        self._init_context(image_dir, neighbor_count, radius, transform, input_mode)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, name = self.base[index]
        curves, distances, mask = self._context(index)
        return NeighborhoodInput(image, curves, distances, mask), name

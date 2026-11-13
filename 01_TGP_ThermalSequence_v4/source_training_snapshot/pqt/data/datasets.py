from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from pqt.data.curve_cache import CurveCache
from pqt.data.sr_cache import load_cached_sr_image
from pqt.model.inputs import DualResolutionInput


def load_image(image_path: Path, input_mode: str) -> Image.Image:
    image = Image.open(image_path)
    if input_mode == "curve":
        # Stripe images replicate one response curve per row; row 0 is the
        # complete signal. Emit it as a 1xW image so downstream transforms
        # (resize to (H, W)) reproduce the original stripe contract.
        row = np.asarray(image.convert("L"))[0]
        return Image.fromarray(row[np.newaxis, :])
    if input_mode == "grayscale":
        return image.convert("L")
    if input_mode == "rgb":
        return image.convert("RGB")
    raise ValueError(f"Unsupported input mode: {input_mode}")


class CsvImageRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        filename_column: str = "Image_Path",
        label_column: str = "Label",
        transform=None,
        input_mode: str = "grayscale",
        sr_image_dir: Path | None = None,
        include_labels: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.filename_column = filename_column
        self.label_column = label_column
        self.transform = transform
        self.input_mode = input_mode
        self.sr_image_dir = Path(sr_image_dir) if sr_image_dir is not None else None
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_name = str(row[self.filename_column])
        image = load_image(self.image_dir / image_name, self.input_mode)
        if self.transform is not None:
            image = self.transform(image)
        if self.sr_image_dir is not None:
            image = DualResolutionInput(image, load_cached_sr_image(self.sr_image_dir / image_name))

        if not self.include_labels:
            return image, image_name

        label = torch.tensor([float(row[self.label_column])], dtype=torch.float32)
        return image, label


def _warp_curve_and_label(
    image: torch.Tensor, label: torch.Tensor, k: float
) -> tuple[torch.Tensor, torch.Tensor]:
    length = image.shape[-1]
    grid = (torch.arange(length, dtype=torch.float32) * k).clamp(0.0, length - 1.0)
    i0 = grid.floor().long()
    i1 = (i0 + 1).clamp(max=length - 1)
    weight = grid - i0.float()
    curve = image.flatten()
    warped = (curve[i0] * (1.0 - weight) + curve[i1] * weight).view(1, 1, -1)
    return warped, label / float(np.sqrt(k))


class CsvCurveRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str]]):
    """CSV-driven dataset backed by a packed curve cache instead of PNG decode.

    Emits the same normalized tensor contract as the image path: a [1, 1, L]
    pseudo-image in [-1, 1], identical to what the stripe image produces after
    ToTensor+Normalize. ``transform`` is accepted for signature parity and is
    expected to be None for curve mode.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        cache: "CurveCache",
        filename_column: str = "Image_Path",
        label_column: str = "Label",
        transform=None,
        include_labels: bool = True,
        time_warp: float = 0.0,
        val_time_warp: float = 0.0,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.cache = cache
        self.filename_column = filename_column
        self.label_column = label_column
        self.transform = transform
        self.include_labels = include_labels
        self.time_warp = float(time_warp)
        self.val_time_warp = float(val_time_warp)
        missing = [
            name for name in self.dataframe[self.filename_column].astype(str) if name not in self.cache.index
        ]
        if missing:
            raise ValueError(f"Curve cache misses {len(missing)} names, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _tensor(self, name: str) -> torch.Tensor:
        values = torch.from_numpy(self.cache.row(name).astype(np.float32))
        return (values / 127.5 - 1.0).view(1, 1, -1)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_name = str(row[self.filename_column])
        image = self._tensor(image_name)
        label = torch.tensor([float(row[self.label_column])], dtype=torch.float32)
        if self.time_warp > 0 and self.include_labels:
            # Thermal diffusion length scales as sqrt(t): resampling the curve
            # at t*k mimics a defect at depth d/sqrt(k). Warp curve and label
            # jointly so the pair stays physically consistent.
            k = float(np.exp(np.random.uniform(-self.time_warp, self.time_warp)))
            image, label = _warp_curve_and_label(image, label, k)
        elif self.val_time_warp > 0 and self.include_labels:
            # Validation-time continuous probe: deterministic per-sample warp
            # pushes labels off the 9-class training grid (label/sqrt(k)),
            # exposing whether predictions track continuously or snap to class
            # levels. The factor is keyed on the filename so every epoch and
            # worker sees the same warp for a given sample.
            rng = np.random.default_rng(zlib.crc32(image_name.encode("utf-8")))
            k = float(np.exp(rng.uniform(-self.val_time_warp, self.val_time_warp)))
            image, label = _warp_curve_and_label(image, label, k)
        if self.transform is not None:
            image = self.transform(image)
        if not self.include_labels:
            return image, image_name
        return image, label


class ImageFolderDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        image_dir: Path,
        valid_extensions: tuple[str, ...],
        transform=None,
        input_mode: str = "grayscale",
        sr_image_dir: Path | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.valid_extensions = tuple(extension.lower() for extension in valid_extensions)
        self.transform = transform
        self.input_mode = input_mode
        self.sr_image_dir = Path(sr_image_dir) if sr_image_dir is not None else None
        self.image_files = sorted(
            path.name
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.valid_extensions
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_name = self.image_files[index]
        image = load_image(self.image_dir / image_name, self.input_mode)
        if self.transform is not None:
            image = self.transform(image)
        if self.sr_image_dir is not None:
            image = DualResolutionInput(image, load_cached_sr_image(self.sr_image_dir / image_name))
        return image, image_name

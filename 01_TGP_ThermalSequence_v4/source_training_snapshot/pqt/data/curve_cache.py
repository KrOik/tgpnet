from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def curve_cache_path(image_dir: Path) -> Path:
    # Derived artifacts must live outside data/: the frozen dataset manifest
    # verifies tree SHA-256, so anything written beside the images breaks the
    # external-evaluation gate.
    image_dir = Path(image_dir)
    tag = f"{image_dir.parent.name}_{image_dir.name}"
    return Path("work/cache/curves") / f"{tag}_curves.npz"


def build_curve_cache(
    image_dir: Path,
    output_path: Path | None = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
) -> Path:
    """Extract row 0 of every stripe image into a compact uint8 npz.

    Stripe images replicate one curve vertically, so row 0 is the complete
    signal. The cache is ~7 MB for the full data1 set versus ~350 MB of PNGs
    and removes PNG decode entirely from the training hot path.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path) if output_path is not None else curve_cache_path(image_dir)
    names = sorted(
        path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not names:
        raise ValueError(f"No images found under {image_dir}")
    curves = np.stack(
        [np.asarray(Image.open(image_dir / name).convert("L"))[0] for name in names]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, names=np.asarray(names), curves=curves)
    return output_path


class CurveCache:
    """Filename -> row lookup over a packed curve matrix (uint8 [N, L])."""

    def __init__(self, path: Path) -> None:
        payload = np.load(Path(path), allow_pickle=False)
        self.names = payload["names"].tolist()
        self.curves = payload["curves"]
        self.index = {name: position for position, name in enumerate(self.names)}
        self.length = int(self.curves.shape[1])

    def row(self, name: str) -> np.ndarray:
        if name not in self.index:
            raise KeyError(f"{name} not present in curve cache")
        return self.curves[self.index[name]]


def load_or_build_curve_cache(image_dir: Path, cache_path: Path | None = None) -> CurveCache:
    path = Path(cache_path) if cache_path is not None else curve_cache_path(image_dir)
    if not path.exists():
        build_curve_cache(image_dir, path)
    return CurveCache(path)

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SR_MODEL_ID = "torchsr-edsr-baseline-x2"
SR_MODEL_VERSION = "1.0.3"
SR_SCALE = 2
SR_INPUT_SIZE = (1024, 1024)
SR_OUTPUT_SIZE = (2048, 2048)
SR_WEIGHT_SIZE = 5_491_291
SR_WEIGHT_SHA256 = "2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1"
SR_WEIGHT_RELATIVE_PATH = Path("artifacts/pretrained/edsr64_x2.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_row(asset_root: Path, role: str, dataset_name: str) -> dict[str, str]:
    path = Path(asset_root) / "manifests" / "datasets.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["role"] == role and row["name"] == dataset_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {role}/{dataset_name} dataset manifest row, got {len(matches)}.")
    return matches[0]


def default_sr_cache_paths(asset_root: Path, role: str, dataset_name: str) -> tuple[Path, Path]:
    row = dataset_manifest_row(asset_root, role, dataset_name)
    key = f"{dataset_name}-{row['tree_sha256'][:12]}-{SR_WEIGHT_SHA256[:12]}"
    root = Path(asset_root) / "work" / "cache" / "sr" / key
    return root / "images", root / "manifest.json"


def load_cached_sr_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.uint16).copy()
    if values.ndim != 2:
        raise ValueError(f"Expected a single-channel 16-bit SR image: {path}")
    tensor = torch.from_numpy(values.astype(np.float32) / 65535.0).unsqueeze(0)
    return tensor.sub(0.5).div(0.5)


def verify_sr_cache(
    image_dir: Path,
    manifest_path: Path,
    source_image_dir: Path,
    expected_names: set[str] | None = None,
) -> dict[str, object]:
    image_dir = Path(image_dir)
    manifest_path = Path(manifest_path)
    source_image_dir = Path(source_image_dir)
    if not image_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SR cache or manifest: {image_dir}, {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "sr_model_id": SR_MODEL_ID,
        "sr_model_version": SR_MODEL_VERSION,
        "sr_weight_sha256": SR_WEIGHT_SHA256,
        "scale": SR_SCALE,
        "input_size": list(SR_INPUT_SIZE),
        "output_size": list(SR_OUTPUT_SIZE),
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise ValueError(f"SR cache {name} mismatch: {payload.get(name)!r} != {expected!r}")
    source_names = {path.name for path in source_image_dir.iterdir() if path.is_file()}
    cached_names = {path.name for path in image_dir.iterdir() if path.is_file()}
    target_names = source_names if expected_names is None else expected_names
    if target_names != cached_names:
        raise ValueError("SR cache filenames do not match the requested source images.")
    if int(payload.get("image_count", -1)) != len(cached_names):
        raise ValueError("SR cache image count does not match its manifest.")
    for name in sorted(cached_names):
        with Image.open(image_dir / name) as image:
            if image.size != SR_OUTPUT_SIZE:
                raise ValueError(f"SR cache image {name} has size {image.size}, expected {SR_OUTPUT_SIZE}.")
    return payload


def write_sr_cache_manifest(
    output_path: Path,
    *,
    dataset_role: str,
    dataset_name: str,
    source_tree_sha256: str,
    image_dir: Path,
) -> dict[str, object]:
    files = sorted(path for path in Path(image_dir).iterdir() if path.is_file())
    aggregate = hashlib.sha256()
    total_size = 0
    for path in files:
        total_size += path.stat().st_size
        aggregate.update(f"{_sha256(path)}  ./{path.name}\n".encode())
    payload: dict[str, object] = {
        "dataset_role": dataset_role,
        "dataset_name": dataset_name,
        "source_tree_sha256": source_tree_sha256,
        "sr_model_id": SR_MODEL_ID,
        "sr_model_version": SR_MODEL_VERSION,
        "sr_weight_sha256": SR_WEIGHT_SHA256,
        "scale": SR_SCALE,
        "input_size": list(SR_INPUT_SIZE),
        "output_size": list(SR_OUTPUT_SIZE),
        "grayscale_conversion": "Rec.601",
        "encoding": "uint16_png",
        "image_count": len(files),
        "size_bytes": total_size,
        "tree_sha256": aggregate.hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload

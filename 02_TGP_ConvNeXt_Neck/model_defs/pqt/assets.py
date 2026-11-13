from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_assets(asset_root: Path, manifest_root: Path | None = None) -> list[str]:
    asset_root = Path(asset_root).resolve()
    manifest_root = Path(manifest_root or asset_root).resolve()
    errors: list[str] = []
    for row in _rows(manifest_root / "manifests" / "datasets.csv"):
        root = asset_root / row["relative_path"]
        metadata = root / row["metadata_file"]
        image_dir = root / "images"
        if not metadata.is_file() or not image_dir.is_dir():
            errors.append(f"{row['name']}: missing metadata or image directory")
            continue
        with metadata.open(newline="", encoding="utf-8") as handle:
            filenames = [item[row["filename_column"]] for item in csv.DictReader(handle)]
        image_names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
        if len(filenames) != len(set(filenames)):
            errors.append(f"{row['name']}: duplicate filenames in metadata")
        if set(filenames) != set(image_names):
            errors.append(f"{row['name']}: metadata and image filenames differ")
        if len(image_names) != int(row["image_count"]):
            errors.append(f"{row['name']}: image count {len(image_names)} != {row['image_count']}")
        files = sorted(path for path in root.rglob("*") if path.is_file())
        aggregate = hashlib.sha256()
        size = 0
        for path in files:
            size += path.stat().st_size
            relative = path.relative_to(root).as_posix()
            aggregate.update(f"{file_sha256(path)}  ./{relative}\n".encode())
        if len(files) != int(row["file_count"]):
            errors.append(f"{row['name']}: file count mismatch")
        if size != int(row["size_bytes"]):
            errors.append(f"{row['name']}: size mismatch")
        if aggregate.hexdigest() != row["tree_sha256"]:
            errors.append(f"{row['name']}: tree SHA-256 mismatch")
    for row in _rows(manifest_root / "manifests" / "models.csv"):
        path = asset_root / row["relative_path"]
        if not path.is_file():
            errors.append(f"{row['model_id']}@{row['version']}: model asset missing")
        elif path.stat().st_size != int(row["size_bytes"]) or file_sha256(path) != row["sha256"]:
            errors.append(f"{row['model_id']}@{row['version']}: model asset identity mismatch")
    return errors

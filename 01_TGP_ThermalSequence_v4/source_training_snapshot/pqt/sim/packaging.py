from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


MAT_REQUIRED_KEYS = (
    "x_coords",
    "y_coords",
    "time_axis",
    "temperature_cube",
    "depth_map",
)


@dataclass(slots=True)
class SimulationExport:
    x_coords: np.ndarray
    y_coords: np.ndarray
    time_axis: np.ndarray
    temperature_cube: np.ndarray
    depth_map: np.ndarray


def load_simulation_export(raw_path: Path) -> SimulationExport:
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported simulation export format: {raw_path.suffix}")

    payload = loadmat(raw_path)
    missing = [key for key in MAT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Missing required MAT keys: {missing}")

    x_coords = _to_1d_float_array(payload["x_coords"])
    y_coords = _to_1d_float_array(payload["y_coords"])
    time_axis = _to_1d_float_array(payload["time_axis"])
    temperature_cube = np.asarray(payload["temperature_cube"], dtype=np.float64)
    depth_map = np.asarray(payload["depth_map"], dtype=np.float64)

    if temperature_cube.ndim != 3:
        raise ValueError(f"temperature_cube must be 3D, got shape {temperature_cube.shape}")
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    expected_grid = (y_coords.size, x_coords.size)
    if temperature_cube.shape[:2] != expected_grid:
        raise ValueError(
            "temperature_cube shape does not match coordinate axes: "
            f"{temperature_cube.shape[:2]} vs {expected_grid}"
        )
    if depth_map.shape != expected_grid:
        raise ValueError(f"depth_map shape does not match coordinate axes: {depth_map.shape} vs {expected_grid}")
    if temperature_cube.shape[2] != time_axis.size:
        raise ValueError(
            "temperature_cube time dimension does not match time_axis: "
            f"{temperature_cube.shape[2]} vs {time_axis.size}"
        )

    return SimulationExport(
        x_coords=x_coords,
        y_coords=y_coords,
        time_axis=time_axis,
        temperature_cube=temperature_cube,
        depth_map=depth_map,
    )


def package_simulation_dataset(
    raw_path: Path,
    output_dir: Path,
    dataset_name: str | None = None,
    *,
    stripe_height: int | None = None,
    include_background: bool = False,
    apply_log: bool = True,
    depth_scale: float = 1000.0,
) -> dict[str, object]:
    export = load_simulation_export(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    active_dataset_name = dataset_name or output_dir.name
    csv_path = output_dir / f"{active_dataset_name}.csv"
    metadata_path = output_dir / f"{active_dataset_name}_metadata.csv"
    summary_path = output_dir / f"{active_dataset_name}_summary.json"

    signal_height = stripe_height or int(export.time_axis.size)
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for y_index in range(export.y_coords.size):
        for x_index in range(export.x_coords.size):
            depth_value = float(export.depth_map[y_index, x_index])
            if not include_background and depth_value <= 0.0:
                continue

            signal = export.temperature_cube[y_index, x_index, :]
            image_name = f"{x_index + 1}_{y_index + 1}.png"
            image_path = image_dir / image_name
            stripe = build_stripe_image(signal, stripe_height=signal_height, apply_log=apply_log)
            Image.fromarray(stripe, mode="L").save(image_path)

            label_value = depth_value * depth_scale
            rows.append({"Image_Path": image_name, "Label": label_value})
            metadata_rows.append(
                {
                    "Image_Path": image_name,
                    "Label": label_value,
                    "X_Index": x_index + 1,
                    "Y_Index": y_index + 1,
                    "X_Coord": float(export.x_coords[x_index]),
                    "Y_Coord": float(export.y_coords[y_index]),
                    "Signal_Min_K": float(np.min(signal)),
                    "Signal_Max_K": float(np.max(signal)),
                }
            )

    dataset_frame = pd.DataFrame(rows)
    if dataset_frame.empty:
        raise ValueError("Simulation export produced zero samples after filtering.")
    dataset_frame.to_csv(csv_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    summary = {
        "raw_path": str(Path(raw_path)),
        "output_dir": str(output_dir),
        "dataset_name": active_dataset_name,
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "image_dir": str(image_dir),
        "num_samples": int(len(dataset_frame)),
        "stripe_height": int(signal_height),
        "time_steps": int(export.time_axis.size),
        "grid_shape": [int(export.y_coords.size), int(export.x_coords.size)],
        "include_background": bool(include_background),
        "apply_log": bool(apply_log),
        "depth_scale": float(depth_scale),
        "label_min": float(dataset_frame["Label"].min()),
        "label_max": float(dataset_frame["Label"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "dataset_name": active_dataset_name,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "summary_path": summary_path,
        "image_dir": image_dir,
        "num_samples": len(dataset_frame),
    }


def build_stripe_image(signal: np.ndarray, *, stripe_height: int, apply_log: bool) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("Signal must contain at least one time step.")
    if stripe_height <= 0:
        raise ValueError(f"stripe_height must be positive, got {stripe_height}")

    normalized = signal.copy()
    normalized -= float(normalized.min())
    if apply_log:
        normalized = np.log1p(normalized)

    peak = float(normalized.max())
    if peak > 0.0:
        normalized = normalized / peak
    else:
        normalized = np.zeros_like(normalized)

    row = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
    return np.tile(row[np.newaxis, :], (stripe_height, 1))


def _to_1d_float_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return array.reshape(-1)

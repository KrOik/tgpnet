from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


FEATURE_NAMES = (
    "mean",
    "std",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "gradient_mean",
    "gradient_std",
)


def image_summary_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float64) / 255.0
    gx = np.diff(values, axis=1).ravel()
    gy = np.diff(values, axis=0).ravel()
    gradient = np.concatenate([gx, gy])
    return np.asarray(
        [
            values.mean(),
            values.std(),
            *np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95)),
            np.abs(gradient).mean(),
            gradient.std(),
        ],
        dtype=np.float64,
    )


def extract_feature_matrix(image_dir: Path, filenames: pd.Series) -> np.ndarray:
    return np.vstack([image_summary_features(Path(image_dir) / str(filename)) for filename in filenames])


def fit_fixed_ridge(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> dict[str, np.ndarray | float]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "alpha": float(alpha)}


def predict_fixed_ridge(model: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return design @ np.asarray(model["coefficients"])


def build_baseline_predictions(
    train_csv: Path,
    train_image_dir: Path,
    prediction_names: pd.Series,
    prediction_image_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = pd.read_csv(train_csv)
    required = {"Image_Path", "Label"}
    if not required.issubset(train.columns):
        raise ValueError(f"Baseline training CSV requires {sorted(required)}.")
    train_features = extract_feature_matrix(train_image_dir, train["Image_Path"])
    target = train["Label"].to_numpy(dtype=np.float64)
    ridge = fit_fixed_ridge(train_features, target, alpha=1.0)
    evaluation_features = extract_feature_matrix(prediction_image_dir, prediction_names)
    predictions = pd.DataFrame(
        {
            "Image_Name": prediction_names.astype(str).to_numpy(),
            "constant_prediction": float(target.mean()),
            "weak_ridge_prediction": predict_fixed_ridge(ridge, evaluation_features),
        }
    )
    metadata = {
        "constant_value": float(target.mean()),
        "ridge_alpha": 1.0,
        "feature_names": list(FEATURE_NAMES),
        "train_sample_size": int(len(train)),
    }
    return predictions, metadata


def compare_model_to_baselines(
    scored_frame: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    tile_size: int = 8,
    repetitions: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scored_frame.merge(baseline_predictions, on="Image_Name", validate="one_to_one")
    if not {"X", "Y"}.issubset(frame.columns):
        raise ValueError("Paired baseline comparison requires X and Y coordinates.")
    frame["model_absolute_error"] = (frame["Predicted_Value"] - frame["Ground_Truth"]).abs()
    for baseline_name, prediction_column in (
        ("constant", "constant_prediction"),
        ("weak_ridge", "weak_ridge_prediction"),
    ):
        frame[f"{baseline_name}_absolute_error"] = (frame[prediction_column] - frame["Ground_Truth"]).abs()
    frame["tile_x"] = ((frame["X"] - frame["X"].min()) // tile_size).astype(int)
    frame["tile_y"] = ((frame["Y"] - frame["Y"].min()) // tile_size).astype(int)
    tiles = [part for _, part in frame.groupby(["tile_x", "tile_y"], sort=True)]
    if len(tiles) < 2:
        raise ValueError("Paired comparison requires at least two occupied coordinate tiles.")

    rng = np.random.default_rng(seed)
    sample_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str | bool]] = []
    for baseline_name, prediction_column in (
        ("constant", "constant_prediction"),
        ("weak_ridge", "weak_ridge_prediction"),
    ):
        point_delta = float((frame["model_absolute_error"] - frame[f"{baseline_name}_absolute_error"]).mean())
        deltas = []
        for repetition in range(repetitions):
            sample = pd.concat(
                [tiles[index] for index in rng.integers(0, len(tiles), size=len(tiles))],
                ignore_index=True,
            )
            delta = float(
                (sample["model_absolute_error"] - sample[f"{baseline_name}_absolute_error"]).mean()
            )
            deltas.append(delta)
            sample_rows.append({"baseline": baseline_name, "repetition": repetition, "delta_mae": delta})
        low, high = np.quantile(deltas, (0.025, 0.975))
        summary_rows.append(
            {
                "baseline": baseline_name,
                "model_mae": float(frame["model_absolute_error"].mean()),
                "baseline_mae": float(frame[f"{baseline_name}_absolute_error"].mean()),
                "delta_mae": point_delta,
                "delta_mae_ci_low": float(low),
                "delta_mae_ci_high": float(high),
                "model_significantly_better": bool(high < 0.0),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_exact_cross_role_duplicates(train_image_dir: Path, external_image_dirs: dict[str, Path]) -> dict[str, int]:
    train_hashes = {file_sha256(path): path.name for path in Path(train_image_dir).iterdir() if path.is_file()}
    counts: dict[str, int] = {}
    collisions: list[str] = []
    for dataset_name, image_dir in external_image_dirs.items():
        count = 0
        for path in Path(image_dir).iterdir():
            if not path.is_file():
                continue
            digest = file_sha256(path)
            if digest in train_hashes:
                count += 1
                collisions.append(f"{dataset_name}:{path.name} == train:{train_hashes[digest]}")
        counts[dataset_name] = count
    if collisions:
        raise ValueError("Exact train/test image duplication detected: " + "; ".join(collisions[:20]))
    return counts

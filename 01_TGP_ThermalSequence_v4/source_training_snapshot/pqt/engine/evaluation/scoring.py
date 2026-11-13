from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from pqt.config.schema import StandardEvaluationDatasetSpec
from pqt.engine.metrics import calculate_regression_metrics


def summarize_prediction_distribution(dataset_name: str, predictions: pd.Series) -> dict[str, float | int | str]:
    predictions = predictions.astype(float)
    return {
        "dataset_name": dataset_name,
        "sample_size": int(predictions.size),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std(ddof=0)),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_p05": float(predictions.quantile(0.05)),
        "prediction_p25": float(predictions.quantile(0.25)),
        "prediction_p50": float(predictions.quantile(0.50)),
        "prediction_p75": float(predictions.quantile(0.75)),
        "prediction_p95": float(predictions.quantile(0.95)),
    }


def load_ground_truth_frame(dataset: StandardEvaluationDatasetSpec) -> pd.DataFrame:
    if dataset.ground_truth_csv is None:
        raise ValueError(f"Ground-truth CSV is required for {dataset.name}.")
    ground_truth_frame = pd.read_csv(dataset.ground_truth_csv)
    filename_column = next(
        (candidate for candidate in ("Image_Name", "Image_Path", "Filename") if candidate in ground_truth_frame.columns),
        None,
    )
    if filename_column is None or "Label" not in ground_truth_frame.columns:
        raise ValueError(f"Unsupported ground-truth CSV for {dataset.name}: {list(ground_truth_frame.columns)}")
    coordinate_columns = [name for name in ("X", "Y") if name in ground_truth_frame.columns]
    normalized = ground_truth_frame[[filename_column, *coordinate_columns, "Label"]].copy()
    normalized[filename_column] = normalized[filename_column].astype(str)
    normalized["Label"] = normalized["Label"].astype(float)
    return normalized.rename(columns={filename_column: "Image_Name", "Label": "Ground_Truth"})


def score_predictions_against_ground_truth(
    dataset_name: str,
    prediction_frame: pd.DataFrame,
    ground_truth_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    predictions = prediction_frame[["Image_Name", "Predicted_Value"]].copy()
    predictions["Image_Name"] = predictions["Image_Name"].astype(str)
    predictions["Predicted_Value"] = predictions["Predicted_Value"].astype(float)
    merged = predictions.merge(ground_truth_frame, how="outer", on="Image_Name", indicator=True)
    if not (merged["_merge"] == "both").all():
        missing_predictions = int((merged["_merge"] == "right_only").sum())
        missing_ground_truth = int((merged["_merge"] == "left_only").sum())
        raise ValueError(
            f"Prediction/GT mismatch for {dataset_name}: missing_predictions={missing_predictions}, "
            f"missing_ground_truth={missing_ground_truth}"
        )
    scored_columns = ["Image_Name", "Predicted_Value", "Ground_Truth", *[name for name in ("X", "Y") if name in merged]]
    scored_frame = merged.loc[:, scored_columns].copy()
    metrics = calculate_regression_metrics(
        torch.tensor(scored_frame["Predicted_Value"].to_numpy(), dtype=torch.float32),
        torch.tensor(scored_frame["Ground_Truth"].to_numpy(), dtype=torch.float32),
    )
    metrics["dataset_name"] = dataset_name
    absolute_error = (scored_frame["Predicted_Value"] - scored_frame["Ground_Truth"]).abs()
    metrics.update(
        {
            "median_absolute_error": float(absolute_error.median()),
            "p95_absolute_error": float(absolute_error.quantile(0.95)),
            "max_absolute_error": float(absolute_error.max()),
        }
    )
    return scored_frame, metrics


def spatial_block_bootstrap(
    scored_frame: pd.DataFrame,
    *,
    tile_size: int = 8,
    repetitions: int = 2000,
    seed: int = 42,
    tolerance: float = 0.08,
) -> tuple[dict[str, float], pd.DataFrame]:
    if not {"X", "Y"}.issubset(scored_frame.columns):
        raise ValueError("Spatial bootstrap requires X and Y coordinates.")
    frame = scored_frame.copy()
    frame["error"] = frame["Predicted_Value"].astype(float) - frame["Ground_Truth"].astype(float)
    frame["absolute_error"] = frame["error"].abs()
    frame["tile_x"] = ((frame["X"].astype(float) - float(frame["X"].min())) // tile_size).astype(int)
    frame["tile_y"] = ((frame["Y"].astype(float) - float(frame["Y"].min())) // tile_size).astype(int)
    tiles = [part for _, part in frame.groupby(["tile_x", "tile_y"], sort=True)]
    if len(tiles) < 2:
        raise ValueError("Spatial bootstrap requires at least two occupied coordinate tiles.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(repetitions):
        sample = pd.concat([tiles[index] for index in rng.integers(0, len(tiles), size=len(tiles))], ignore_index=True)
        error = sample["error"].to_numpy(dtype=float)
        absolute = np.abs(error)
        rows.append(
            {
                "mae": float(absolute.mean()),
                "rmse": float(math.sqrt(np.square(error).mean())),
                "median_absolute_error": float(np.median(absolute)),
                "p95_absolute_error": float(np.quantile(absolute, 0.95)),
                "bias": float(error.mean()),
                "tolerance_coverage": float((absolute <= tolerance).mean()),
            }
        )
    samples = pd.DataFrame(rows)
    intervals: dict[str, float] = {"tile_count": float(len(tiles))}
    for name in samples.columns:
        intervals[f"{name}_ci_low"] = float(samples[name].quantile(0.025))
        intervals[f"{name}_ci_high"] = float(samples[name].quantile(0.975))
    return intervals, samples


def paired_model_block_bootstrap(
    scored_frames: dict[str, pd.DataFrame],
    *,
    candidate_name: str,
    tile_size: int = 8,
    repetitions: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models with exactly the same spatial block resamples."""
    if candidate_name not in scored_frames or len(scored_frames) < 2:
        raise ValueError("Paired model bootstrap requires a candidate and at least one comparator.")
    base: pd.DataFrame | None = None
    for model_name, frame in scored_frames.items():
        required = {"Image_Name", "Predicted_Value", "Ground_Truth", "X", "Y"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing paired-bootstrap columns for {model_name}.")
        current = frame[list(required)].copy().sort_values("Image_Name").reset_index(drop=True)
        current = current.rename(columns={"Predicted_Value": f"prediction::{model_name}"})
        if base is None:
            base = current
        else:
            base = base.merge(current, on=["Image_Name", "Ground_Truth", "X", "Y"], validate="one_to_one")
    assert base is not None
    base["tile_x"] = ((base["X"] - base["X"].min()) // tile_size).astype(int)
    base["tile_y"] = ((base["Y"] - base["Y"].min()) // tile_size).astype(int)
    tiles = [part.index.to_numpy() for _, part in base.groupby(["tile_x", "tile_y"], sort=True)]
    if len(tiles) < 2:
        raise ValueError("Paired model bootstrap requires at least two occupied coordinate tiles.")
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([tiles[index] for index in rng.integers(0, len(tiles), len(tiles))]) for _ in range(repetitions)]
    target = base["Ground_Truth"].to_numpy(dtype=float)
    errors = {
        name: base[f"prediction::{name}"].to_numpy(dtype=float) - target
        for name in scored_frames
    }
    rows: list[dict[str, float | str | int]] = []
    summaries: list[dict[str, float | str]] = []
    candidate_error = errors[candidate_name]
    for comparator, comparator_error in errors.items():
        if comparator == candidate_name:
            continue
        delta_mae_samples: list[float] = []
        delta_rmse_samples: list[float] = []
        for repetition, draw in enumerate(draws):
            candidate_draw = candidate_error[draw]
            comparator_draw = comparator_error[draw]
            delta_mae = float(np.abs(candidate_draw).mean() - np.abs(comparator_draw).mean())
            delta_rmse = float(np.sqrt(np.square(candidate_draw).mean()) - np.sqrt(np.square(comparator_draw).mean()))
            delta_mae_samples.append(delta_mae)
            delta_rmse_samples.append(delta_rmse)
            rows.append({"candidate": candidate_name, "comparator": comparator, "repetition": repetition,
                         "delta_mae": delta_mae, "delta_rmse": delta_rmse})
        mae_low, mae_high = np.quantile(delta_mae_samples, (0.025, 0.975))
        rmse_low, rmse_high = np.quantile(delta_rmse_samples, (0.025, 0.975))
        summaries.append({
            "candidate": candidate_name,
            "comparator": comparator,
            "candidate_mae": float(np.abs(candidate_error).mean()),
            "comparator_mae": float(np.abs(comparator_error).mean()),
            "delta_mae": float(np.abs(candidate_error).mean() - np.abs(comparator_error).mean()),
            "delta_mae_ci_low": float(mae_low), "delta_mae_ci_high": float(mae_high),
            "candidate_rmse": float(np.sqrt(np.square(candidate_error).mean())),
            "comparator_rmse": float(np.sqrt(np.square(comparator_error).mean())),
            "delta_rmse": float(np.sqrt(np.square(candidate_error).mean()) - np.sqrt(np.square(comparator_error).mean())),
            "delta_rmse_ci_low": float(rmse_low), "delta_rmse_ci_high": float(rmse_high),
        })
    return pd.DataFrame(summaries), pd.DataFrame(rows)

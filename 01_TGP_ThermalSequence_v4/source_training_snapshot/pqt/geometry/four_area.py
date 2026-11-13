from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


XRANGE = (228, 263)
YRANGE = (361, 394)
COUNT = 1224
REFERENCE_DEPTH_POSITIONS = np.array([62.0 / 255.0, 95.0 / 255.0, 165.0 / 255.0, 241.0 / 255.0], dtype=np.float64)
SMOOTHING_CANDIDATES = np.array([0.0, 0.35, 0.7, 1.05, 1.4, 1.75, 2.1], dtype=np.float64)
DEPTH_POSITION_PERMUTATIONS = tuple(np.asarray(item, dtype=np.float64) for item in itertools.permutations(REFERENCE_DEPTH_POSITIONS.tolist()))


@dataclass(frozen=True)
class FitResult:
    split_x: float
    split_y: float
    smoothing_width_px: float
    plateau_depths: np.ndarray
    rmse: float
    mse: float
    reconstructed_grid: np.ndarray
    region_ids: np.ndarray


def load_prediction_frame(csvPath: Path) -> pd.DataFrame:
    csvPath = Path(csvPath)
    if not csvPath.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csvPath}")
    frame = pd.read_csv(csvPath)
    frame = normalize_prediction_columns(frame)
    validate_grid(frame)
    return frame


def normalize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    filenameCandidates = ("Filename", "Image_Path", "Image_Name")
    predictionCandidates = ("Predicted", "Predicted_Value", "Prediction", "predicted")

    nameCol = next((name for name in filenameCandidates if name in frame.columns), None)
    predCol = next((name for name in predictionCandidates if name in frame.columns), None)
    if nameCol is None or predCol is None:
        raise KeyError(f"Unsupported prediction columns: {list(frame.columns)}")

    rename[nameCol] = "Filename"
    rename[predCol] = "Predicted"
    frame = frame.rename(columns=rename)

    if "X" not in frame.columns or "Y" not in frame.columns:
        xy = frame["Filename"].str.replace(".png", "", regex=False).str.split("_", expand=True).astype(int)
        frame["X"] = xy[0]
        frame["Y"] = xy[1]

    keep = ["Filename", "X", "Y", "Predicted"]
    frame = frame.loc[:, keep].copy()
    frame["Filename"] = frame["Filename"].astype(str)
    frame["X"] = frame["X"].astype(int)
    frame["Y"] = frame["Y"].astype(int)
    frame["Predicted"] = frame["Predicted"].astype(float)
    frame = frame.sort_values(["Y", "X"], kind="stable").reset_index(drop=True)
    return frame


def validate_grid(frame: pd.DataFrame) -> None:
    xs = sorted(frame["X"].unique().tolist())
    ys = sorted(frame["Y"].unique().tolist())
    if not xs or not ys:
        raise ValueError("Prediction grid is empty.")
    if (xs[0], xs[-1]) != XRANGE:
        raise ValueError(f"Bad x range: {(xs[0], xs[-1])}")
    if (ys[0], ys[-1]) != YRANGE:
        raise ValueError(f"Bad y range: {(ys[0], ys[-1])}")
    if len(frame) != COUNT:
        raise ValueError(f"Bad count: {len(frame)}")
    if len(xs) * len(ys) != len(frame):
        raise ValueError("Grid is not full.")
    if frame["Filename"].duplicated().any():
        raise ValueError("Duplicate filenames detected.")
    if frame[["X", "Y"]].duplicated().any():
        raise ValueError("Duplicate coordinates detected.")


def fit_four_area_geometry(frame: pd.DataFrame) -> FitResult:
    grid = frame.pivot(index="Y", columns="X", values="Predicted").sort_index().sort_index(axis=1)
    ys = grid.index.to_numpy(dtype=np.float64)
    xs = grid.columns.to_numpy(dtype=np.float64)
    observed = grid.to_numpy(dtype=np.float64)
    flatObserved = observed.reshape(-1)

    best: FitResult | None = None
    for xIndex in range(2, len(xs) - 2):
        split_x = float(xs[xIndex - 1])
        for yIndex in range(2, len(ys) - 2):
            split_y = float(ys[yIndex - 1])
            for width in SMOOTHING_CANDIDATES:
                weights = quadrant_weights(xs, ys, split_x, split_y, float(width))
                design = np.column_stack([item.reshape(-1) for item in weights])
                regionIds = np.argmax(np.stack(weights, axis=2), axis=2) + 1
                for depthPositions in DEPTH_POSITION_PERMUTATIONS:
                    template = design @ depthPositions
                    affineDesign = np.column_stack([np.ones_like(template), template])
                    offsetScale, _, _, _ = np.linalg.lstsq(affineDesign, flatObserved, rcond=None)
                    offset = float(offsetScale[0])
                    scale = float(offsetScale[1])
                    if scale <= 0.0:
                        continue
                    depths = offset + scale * depthPositions
                    reconstructed = (design @ depths).reshape(observed.shape)
                    residual = observed - reconstructed
                    mse = float(np.mean(residual * residual))
                    rmse = math.sqrt(mse)
                    candidate = FitResult(
                        split_x=split_x,
                        split_y=split_y,
                        smoothing_width_px=float(width),
                        plateau_depths=np.asarray(depths, dtype=np.float64),
                        rmse=rmse,
                        mse=mse,
                        reconstructed_grid=reconstructed,
                        region_ids=regionIds.astype(int),
                    )
                    if best is None or candidate.mse < best.mse:
                        best = candidate

    if best is None:
        raise RuntimeError("Unable to fit four-area reconstruction.")
    return best


def quadrant_weights(
    xs: np.ndarray,
    ys: np.ndarray,
    split_x: float,
    split_y: float,
    smoothing_width_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xEdge = split_x + 0.5
    yEdge = split_y + 0.5
    left = logistic_left(xs, xEdge, smoothing_width_px)
    top = logistic_left(ys, yEdge, smoothing_width_px)

    tl = np.outer(top, left)
    tr = np.outer(top, 1.0 - left)
    bl = np.outer(1.0 - top, left)
    br = np.outer(1.0 - top, 1.0 - left)
    return tl, tr, bl, br


def logistic_left(values: np.ndarray, edge: float, width: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 0.0:
        return (values < edge).astype(np.float64)
    scaled = (values - edge) / max(width, 1e-6)
    return 1.0 / (1.0 + np.exp(scaled))


def make_reconstructed_frame(frame: pd.DataFrame, fit: FitResult) -> pd.DataFrame:
    reconstructed = frame.loc[:, ["Filename", "X", "Y"]].copy()
    reconstructed["Label"] = fit.reconstructed_grid.reshape(-1)
    return reconstructed

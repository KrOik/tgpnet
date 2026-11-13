from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlaneFit:
    x_center: float
    y_center: float
    base_depth: float
    slope_x: float
    slope_y: float
    clip_min: float
    clip_max: float
    r2: float
    rmse: float


def importPrediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weak observation CSV missing: {path}")
    frame = pd.read_csv(path)
    need = {"Filename", "X", "Y", "Predicted"}
    miss = need.difference(frame.columns)
    if miss:
        raise KeyError(f"Missing prediction columns: {sorted(miss)}")
    out = frame.loc[:, ["Filename", "X", "Y", "Predicted"]].copy()
    out["Filename"] = out["Filename"].astype(str)
    out["X"] = out["X"].astype(int)
    out["Y"] = out["Y"].astype(int)
    out["Predicted"] = out["Predicted"].astype(float)
    validateGrid(out)
    return out.sort_values(["Y", "X"], kind="stable").reset_index(drop=True)


def validateGrid(frame: pd.DataFrame) -> None:
    if frame["Filename"].duplicated().any():
        raise ValueError("Weak observation has duplicate filenames.")
    if frame[["X", "Y"]].duplicated().any():
        raise ValueError("Weak observation has duplicate coordinates.")
    xs = sorted(frame["X"].unique().tolist())
    ys = sorted(frame["Y"].unique().tolist())
    if len(frame) != len(xs) * len(ys):
        raise ValueError("Weak observation grid is incomplete.")


def fitPlane(frame: pd.DataFrame) -> PlaneFit:
    x = frame["X"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    z = frame["Predicted"].to_numpy(dtype=float)
    x_center = float(x.mean())
    y_center = float(y.mean())
    centered_x = x - x_center
    centered_y = y - y_center
    design = np.column_stack([np.ones_like(centered_x), centered_x, centered_y])
    coef, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    fitted = design @ coef
    residual = z - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    return PlaneFit(
        x_center=x_center,
        y_center=y_center,
        base_depth=float(coef[0]),
        slope_x=float(coef[1]),
        slope_y=float(coef[2]),
        clip_min=float(z.min()),
        clip_max=float(z.max()),
        r2=float(r2),
        rmse=rmse,
    )


def buildReconstructed(frame: pd.DataFrame, fit: PlaneFit) -> pd.DataFrame:
    dx = frame["X"].to_numpy(dtype=float) - fit.x_center
    dy = frame["Y"].to_numpy(dtype=float) - fit.y_center
    values = fit.base_depth + fit.slope_x * dx + fit.slope_y * dy
    values = np.clip(values, fit.clip_min, fit.clip_max)
    return pd.DataFrame(
        {
            "Filename": frame["Filename"],
            "X": frame["X"],
            "Y": frame["Y"],
            "Label": values.astype(float),
        }
    )

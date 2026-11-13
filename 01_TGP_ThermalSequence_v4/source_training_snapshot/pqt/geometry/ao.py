from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def readPrediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    df = pd.read_csv(path)
    need = {"Filename", "X", "Y", "Predicted"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise KeyError(f"Missing prediction columns: {miss}")
    out = df.loc[:, ["Filename", "X", "Y", "Predicted"]].copy()
    out["Filename"] = out["Filename"].astype(str)
    out["X"] = out["X"].astype(float)
    out["Y"] = out["Y"].astype(float)
    out["Predicted"] = out["Predicted"].astype(float)
    if out["Filename"].duplicated().any():
        raise ValueError("Duplicate filenames in weak observation.")
    if out[["X", "Y"]].duplicated().any():
        raise ValueError("Duplicate coordinates in weak observation.")
    xs = np.sort(out["X"].unique())
    ys = np.sort(out["Y"].unique())
    if len(xs) * len(ys) != len(out):
        raise ValueError("Weak observation is not a full AO grid.")
    return out.sort_values(["X", "Y"], kind="stable").reset_index(drop=True)


def fitStripModel(df: pd.DataFrame) -> dict[str, float]:
    profile = df.groupby("Y", sort=True)["Predicted"].mean().reset_index()
    yVals = profile["Y"].to_numpy(dtype=float)
    zVals = profile["Predicted"].to_numpy(dtype=float)

    baseline = float(np.median(zVals[(yVals <= np.quantile(yVals, 0.2)) | (yVals >= np.quantile(yVals, 0.8))]))
    groove = float(np.median(np.sort(zVals)[: max(3, len(zVals) // 7)]))

    weights = np.clip(baseline - zVals, 0.0, None)
    if float(np.sum(weights)) <= 0.0:
        centerline = float(yVals[np.argmin(zVals)])
    else:
        centerline = float(np.sum(yVals * weights) / np.sum(weights))

    midpoint = groove + 0.5 * (baseline - groove)
    inside = yVals[zVals <= midpoint]
    grooveWidth = float(inside.max() - inside.min()) if inside.size else 10.0
    grooveWidth = max(2.0, grooveWidth)

    lowQuarter = groove + 0.25 * (baseline - groove)
    highQuarter = groove + 0.75 * (baseline - groove)
    inner = yVals[zVals <= lowQuarter]
    outer = yVals[zVals <= highQuarter]
    if inner.size and outer.size:
        edgeGap = max(0.0, (outer.max() - outer.min()) - (inner.max() - inner.min()))
        transition = max(0.8, edgeGap / 4.0)
    else:
        transition = 1.5

    xStats = []
    xs = np.sort(df["X"].unique())
    for x in xs:
        column = df.loc[df["X"] == x].sort_values("Y", kind="stable")
        cy = column["Y"].to_numpy(dtype=float)
        cz = column["Predicted"].to_numpy(dtype=float)
        cweights = np.clip(baseline - cz, 0.0, None)
        centerX = centerline if float(np.sum(cweights)) <= 0.0 else float(np.sum(cy * cweights) / np.sum(cweights))
        insideBand = np.abs(cy - centerX) <= max(2.0, grooveWidth / 3.0)
        outsideBand = np.abs(cy - centerX) >= (grooveWidth / 2.0 + transition)
        grooveX = float(np.median(cz[insideBand])) if np.any(insideBand) else groove
        baselineX = float(np.median(cz[outsideBand])) if np.any(outsideBand) else baseline
        threshold = grooveX + 0.5 * (baselineX - grooveX)
        widthBand = cy[cz <= threshold]
        widthX = float(widthBand.max() - widthBand.min()) if widthBand.size else grooveWidth
        xStats.append(
            {
                "X": float(x),
                "centerline_y": centerX,
                "baseline_depth": baselineX,
                "groove_depth": grooveX,
                "groove_width": max(2.0, widthX),
            }
        )

    xTable = pd.DataFrame(xStats)
    xn = normalize(xTable["X"].to_numpy(dtype=float))
    centerSlope, centerIntercept = fitLine(xn, xTable["centerline_y"].to_numpy(dtype=float))
    baselineSlope, baselineIntercept = fitLine(xn, xTable["baseline_depth"].to_numpy(dtype=float))
    grooveSlope, grooveIntercept = fitLine(xn, xTable["groove_depth"].to_numpy(dtype=float))
    widthSlope, widthIntercept = fitLine(xn, xTable["groove_width"].to_numpy(dtype=float))

    return {
        "centerline_y": centerline,
        "groove_width": grooveWidth,
        "baseline_depth": baseline,
        "groove_depth": groove,
        "transition_width": transition,
        "centerline_intercept": centerIntercept,
        "centerline_slope_x": centerSlope,
        "baseline_intercept": baselineIntercept,
        "baseline_slope_x": baselineSlope,
        "groove_intercept": grooveIntercept,
        "groove_slope_x": grooveSlope,
        "width_intercept": widthIntercept,
        "width_slope_x": widthSlope,
    }


def makeSurface(df: pd.DataFrame, fit: dict[str, float]) -> pd.DataFrame:
    work = df.loc[:, ["Filename", "X", "Y"]].copy()
    xn = normalize(work["X"].to_numpy(dtype=float))
    y = work["Y"].to_numpy(dtype=float)

    center = fit["centerline_intercept"] + fit["centerline_slope_x"] * xn
    width = fit["width_intercept"] + fit["width_slope_x"] * xn
    baseline = fit["baseline_intercept"] + fit["baseline_slope_x"] * xn
    groove = fit["groove_intercept"] + fit["groove_slope_x"] * xn
    transition = fit["transition_width"]

    width = np.clip(width, 2.0, None)
    groove = np.minimum(groove, baseline - 0.02)
    blend = smoothBlend(np.abs(y - center), width / 2.0, transition)
    label = groove + (baseline - groove) * blend

    return pd.DataFrame(
        {
            "Filename": work["Filename"],
            "X": work["X"].astype(float),
            "Y": work["Y"].astype(float),
            "Label": label.astype(float),
        }
    )


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def fitLine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    return float(slope), float(intercept)


def smoothBlend(distance: np.ndarray, halfWidth: np.ndarray | float, transition: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((np.asarray(distance, dtype=float) - np.asarray(halfWidth, dtype=float)) / max(float(transition), 1e-6)))

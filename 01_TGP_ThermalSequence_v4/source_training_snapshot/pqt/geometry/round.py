from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


CENTER_HINT = (245.25, 431.15)


def load_prediction_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    need = {"Filename", "X", "Y", "Predicted"}
    miss = sorted(need - set(frame.columns))
    if miss:
        raise KeyError(f"Prediction CSV missing columns: {miss}")
    out = frame.loc[:, ["Filename", "X", "Y", "Predicted"]].copy()
    out["X"] = out["X"].round().astype(int)
    out["Y"] = out["Y"].round().astype(int)
    out["Predicted"] = out["Predicted"].astype(float)
    out = out.sort_values(["Y", "X"], kind="stable").reset_index(drop=True)
    check_prediction_grid(out)
    return out


def check_prediction_grid(frame: pd.DataFrame) -> None:
    if len(frame) != 1793:
        raise ValueError(f"Unexpected point count: {len(frame)}")
    if frame[["X", "Y"]].duplicated().any():
        raise ValueError("Duplicate coordinates detected in prediction CSV.")
    if int(frame["X"].min()) != 221 or int(frame["X"].max()) != 269:
        raise ValueError("Unexpected X range for sample_round.")
    if int(frame["Y"].min()) != 407 or int(frame["Y"].max()) != 455:
        raise ValueError("Unexpected Y range for sample_round.")


def fit_round_model(frame: pd.DataFrame) -> dict[str, object]:
    x = frame["X"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    z = frame["Predicted"].to_numpy(dtype=float)

    inverse = (float(np.max(z)) - z) + 1e-6
    weighted_center = (
        float(np.average(x, weights=inverse)),
        float(np.average(y, weights=inverse)),
    )
    center_seed = ((weighted_center[0] + CENTER_HINT[0]) / 2.0, (weighted_center[1] + CENTER_HINT[1]) / 2.0)

    candidates: list[tuple[float, float]] = []
    for base in (weighted_center, CENTER_HINT, center_seed):
        for dx in (-0.5, 0.0, 0.5):
            for dy in (-0.5, 0.0, 0.5):
                candidates.append((round(base[0] + dx, 3), round(base[1] + dy, 3)))
    candidates = list(dict.fromkeys(candidates))

    best: dict[str, object] | None = None
    for cx, cy in candidates:
        candidate = fit_zones_for_center(x, y, z, cx, cy)
        if best is None or float(candidate["rmse"]) < float(best["rmse"]):
            best = candidate
    assert best is not None
    best["center_hint_xy"] = [CENTER_HINT[0], CENTER_HINT[1]]
    best["model_family"] = "three_zone_concentric_round"
    return best


def fit_zones_for_center(x: np.ndarray, y: np.ndarray, z: np.ndarray, cx: float, cy: float) -> dict[str, object]:
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    order = np.argsort(radius, kind="stable")
    r = radius[order]
    v = z[order]
    n = len(r)
    min_seg = max(120, n // 16)
    step = max(6, n // 200)

    prefix = np.concatenate([[0.0], np.cumsum(v)])
    prefix2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def segment(i: int, j: int) -> tuple[float, float]:
        total = prefix[j] - prefix[i]
        total2 = prefix2[j] - prefix2[i]
        mean = total / (j - i)
        error = total2 - 2.0 * mean * total + (j - i) * mean * mean
        return float(error), float(mean)

    best: tuple[float, int, int, float, float, float] | None = None
    for i in range(min_seg, n - 2 * min_seg, step):
        for j in range(i + min_seg, n - min_seg, step):
            err1, m1 = segment(0, i)
            err2, m2 = segment(i, j)
            err3, m3 = segment(j, n)
            if not (m1 <= m2 <= m3):
                continue
            error = err1 + err2 + err3
            if best is None or error < best[0]:
                best = (float(error), i, j, m1, m2, m3)
    if best is None:
        q1 = int(round(n * 0.18))
        q2 = int(round(n * 0.62))
        err1, m1 = segment(0, q1)
        err2, m2 = segment(q1, q2)
        err3, m3 = segment(q2, n)
        best = (float(err1 + err2 + err3), q1, q2, m1, m2, m3)

    _, i, j, m1, m2, m3 = best
    inner_radius = float(r[i - 1])
    middle_radius = float(r[j - 1])
    transition_width = float(max(0.85, min(2.6, (middle_radius - inner_radius) / 6.5)))
    smoothed = smooth_profile(radius, inner_radius, middle_radius, [m1, m2, m3], transition_width)
    rmse = float(np.sqrt(np.mean((smoothed - z) ** 2)))

    return {
        "center_xy": [float(cx), float(cy)],
        "inner_radius": inner_radius,
        "middle_radius": middle_radius,
        "depths": {
            "inner": float(m1),
            "middle": float(m2),
            "outer": float(m3),
        },
        "transition_width": transition_width,
        "rmse": rmse,
    }


def build_label_frame(
    source: pd.DataFrame,
    center_xy: list[float],
    inner_radius: float,
    middle_radius: float,
    depths: dict[str, float],
    transition_width: float,
) -> pd.DataFrame:
    radius = np.sqrt((source["X"].to_numpy(dtype=float) - center_xy[0]) ** 2 + (source["Y"].to_numpy(dtype=float) - center_xy[1]) ** 2)
    label = smooth_profile(
        radius,
        inner_radius,
        middle_radius,
        [float(depths["inner"]), float(depths["middle"]), float(depths["outer"])],
        transition_width,
    )
    out = source.loc[:, ["Filename", "X", "Y"]].copy()
    out["Label"] = label
    return out


def smooth_profile(radius: np.ndarray, inner_radius: float, middle_radius: float, depths: list[float], transition_width: float) -> np.ndarray:
    width = max(0.25, float(transition_width))
    s1 = 1.0 / (1.0 + np.exp(-(radius - inner_radius) / width))
    s2 = 1.0 / (1.0 + np.exp(-(radius - middle_radius) / width))
    return depths[0] + (depths[1] - depths[0]) * s1 + (depths[2] - depths[1]) * s2

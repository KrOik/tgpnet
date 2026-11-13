from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Mapping


DEPTH_SLICES = {
    "shallow": (0.24, 0.40, 0.56),
    "middle": (0.72, 0.88, 1.04),
    "deep": (1.20, 1.36, 1.52),
}


def half_up_median_epoch(best_epochs: Iterable[int]) -> int:
    values = [int(value) for value in best_epochs]
    if not values or any(value < 1 for value in values):
        raise ValueError("best epochs must be a non-empty sequence of positive integers.")
    return int(math.floor(float(median(values)) + 0.5))


def select_robust_loss(rows: Iterable[Mapping[str, float | str]]) -> dict[str, float | str]:
    candidates = [dict(row) for row in rows if row.get("loss_name") in {"huber", "quantile"}]
    if not candidates:
        raise ValueError("Robust-loss selection requires Huber or quantile candidates.")

    def key(row: dict[str, float | str]) -> tuple[float, float, float, int, float]:
        name = str(row["loss_name"])
        parameter = float(row["huber_delta"] if name == "huber" else row["quantile_tau"])
        return (
            float(row["mae"]),
            float(row["rmse"]),
            abs(float(row["bias"])),
            0 if name == "huber" else 1,
            parameter,
        )

    return min(candidates, key=key)


def select_depth_weight_gamma(rows: Iterable[Mapping[str, float]]) -> dict[str, float | bool | str]:
    candidates = [dict(row) for row in rows]
    baselines = [row for row in candidates if float(row["gamma"]) == 0.0]
    if len(baselines) != 1:
        raise ValueError("Gamma selection requires exactly one gamma=0 baseline.")
    baseline = baselines[0]
    qualified = [
        row for row in candidates
        if float(row["gamma"]) > 0.0
        and float(row["deep_mae"]) < float(baseline["deep_mae"])
        and float(row["shallow_mae"]) <= 1.02 * float(baseline["shallow_mae"])
    ]
    if not qualified:
        return {**baseline, "qualified": False, "weighting_status": "inconclusive"}
    selected = min(
        qualified,
        key=lambda row: (
            float(row["mae"]), float(row.get("rmse", float("inf"))),
            abs(float(row.get("bias", float("inf")))), float(row["gamma"]),
        ),
    )
    return {**selected, "qualified": True, "weighting_status": "selected"}


def promotion_decision(
    per_geometry_metrics: Iterable[Mapping[str, float | str]],
    paired_intervals: Iterable[Mapping[str, float | str]],
) -> dict[str, object]:
    metrics = [dict(row) for row in per_geometry_metrics]
    intervals = [dict(row) for row in paired_intervals]
    geometries = sorted({str(row["dataset_name"]) for row in metrics})
    required_models = {"candidate", "v0.3.0", "efficientnet_v2_s", "resnet50"}
    failures: list[str] = []
    for geometry in geometries:
        by_model = {str(row["model_name"]): row for row in metrics if str(row["dataset_name"]) == geometry}
        if not required_models.issubset(by_model):
            failures.append(f"{geometry}:missing_metrics")
            continue
        candidate = by_model["candidate"]
        learned = [by_model[name] for name in ("v0.3.0", "efficientnet_v2_s", "resnet50")]
        if not all(float(candidate[metric]) < float(other[metric]) for metric in ("mae", "rmse") for other in learned):
            failures.append(f"{geometry}:point_metrics")
        best_learned = min(learned, key=lambda row: float(row["mae"]))
        required_comparators = {str(best_learned["model_name"]), "constant", "weak_ridge"}
        comparison = {
            str(row["comparator"]): row
            for row in intervals
            if str(row["dataset_name"]) == geometry and str(row.get("candidate", "candidate")) == "candidate"
        }
        for comparator in required_comparators:
            row = comparison.get(comparator)
            if row is None or float(row["delta_mae_ci_high"]) >= 0.0:
                failures.append(f"{geometry}:paired_mae:{comparator}")
    return {"promote": not failures and bool(geometries), "failures": failures, "geometries": geometries}

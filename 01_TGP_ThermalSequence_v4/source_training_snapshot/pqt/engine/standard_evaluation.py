from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pqt.config.schema import DataConfig, EvaluationConfig, RuntimeConfig, StandardEvaluationDatasetSpec
from pqt.assets import verify_assets
from pqt.data.loaders import create_image_prediction_loader, create_supervised_dataloaders
from pqt.model.checkpoint import load_checkpoint
from pqt.model.inputs import move_model_input
from pqt.model.factory import build_model
from pqt.experiments.contracts import ExperimentRuntime
from pqt.experiments.loader import apply_experiment_model_config, resolve_experiment
from pqt.render.matlab import run_matlab_render
from pqt.viz.evaluation import plot_evaluation_results, write_metrics_summary
from pqt.viz.external_probe import plot_prediction_histogram

from .evaluation.scoring import (
    load_ground_truth_frame,
    score_predictions_against_ground_truth,
    spatial_block_bootstrap,
    summarize_prediction_distribution,
)
from .evaluation.baselines import (
    assert_no_exact_cross_role_duplicates,
    build_baseline_predictions,
    compare_model_to_baselines,
    file_sha256,
)

from .metrics import calculate_regression_metrics
from .runtime import resolve_device, seed_runtime


CANONICAL_EXTERNAL_DATASETS = ("sample_4areas", "sample_ao", "sample_round", "sample_xiemian")


def _require_canonical_frozen_assets(config: EvaluationConfig) -> dict[str, str]:
    if config.data.csv_path is None or config.data.image_dir is None or config.checkpoint_path is None:
        raise ValueError("Frozen evaluation requires canonical training data and checkpoint paths.")
    training_csv = Path(config.data.csv_path).resolve()
    if training_csv.name != "data1.csv" or training_csv.parent.name != "data1":
        raise ValueError("Frozen evaluation requires data/train/data1/data1.csv.")
    asset_root = training_csv.parents[3]
    expected_train_csv = asset_root / "data" / "train" / "data1" / "data1.csv"
    expected_train_images = asset_root / "data" / "train" / "data1" / "images"
    if training_csv != expected_train_csv.resolve() or Path(config.data.image_dir).resolve() != expected_train_images.resolve():
        raise ValueError("Frozen evaluation training-data paths do not match canonical data1 assets.")

    names = tuple(dataset.name for dataset in config.external_datasets)
    if names != CANONICAL_EXTERNAL_DATASETS:
        raise ValueError(f"Frozen evaluation requires exactly {CANONICAL_EXTERNAL_DATASETS}, got {names}.")
    for dataset in config.external_datasets:
        expected_root = asset_root / "data" / "eval" / dataset.name
        if Path(dataset.image_dir).resolve() != (expected_root / "images").resolve():
            raise ValueError(f"Non-canonical external image path for {dataset.name}.")
        if dataset.ground_truth_csv is None or Path(dataset.ground_truth_csv).resolve() != (expected_root / "gt.csv").resolve():
            raise ValueError(f"Non-canonical external ground truth for {dataset.name}.")

    repo_root = Path(__file__).resolve().parents[2]
    active_manifest_path = repo_root / "manifests" / "active_experiment.json"
    registered_paths: set[Path] = set()
    if active_manifest_path.is_file():
        active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))
        registered_checkpoint = active_manifest.get("checkpoint") if isinstance(active_manifest, dict) else None
        if registered_checkpoint:
            registered_paths.add((asset_root / str(registered_checkpoint)).resolve())
    models_manifest_path = repo_root / "manifests" / "models.csv"
    with models_manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("role") == "candidate":
                registered_paths.add((asset_root / row["relative_path"]).resolve())
    if Path(config.checkpoint_path).resolve() not in registered_paths:
        raise ValueError("Frozen evaluation is manifest-locked: checkpoint is not the active model or a registered candidate entry.")
    # A standard external evaluation consumes only canonical data1 plus the
    # four declared sample geometries.  Optional archival external datasets
    # may legitimately be unavailable on a compute node and must not block
    # this frozen protocol.
    required_dataset_names = {"data1", *names}
    errors = [
        error for error in verify_assets(asset_root, repo_root)
        if error.split(":", 1)[0] in required_dataset_names or "@" in error
    ]
    if errors:
        raise ValueError("Frozen asset verification failed: " + "; ".join(errors))
    return {}

def run_standard_evaluation(
    config: EvaluationConfig,
    experiment: ExperimentRuntime | None = None,
) -> dict[str, object]:
    config = deepcopy(config)
    if config.checkpoint_path is None:
        raise ValueError("Evaluation requires checkpoint_path.")
    model_identity = _require_canonical_frozen_assets(config)

    experiment = experiment or resolve_experiment(config.experiment)
    config.model = apply_experiment_model_config(config.model, experiment)
    seed_runtime(config.split.seed, config.runtime.deterministic)

    device = resolve_device(config.runtime)
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config.model, experiment=experiment).to(device)
    metadata = load_checkpoint(config.checkpoint_path, model=model, map_location=device, strict=True)
    checkpoint_config = metadata.get("config") if isinstance(metadata, dict) else None
    if isinstance(checkpoint_config, dict):
        split_config = checkpoint_config.get("split", {})
        selection_config = checkpoint_config.get("best_checkpoint", {})
        if split_config.get("strategy") != "full" or selection_config.get("strategy") not in {"fixed_epochs", "train_loss"}:
            raise ValueError("External evaluation only accepts a frozen full-data checkpoint selected without external data.")
    model.eval()

    results: dict[str, object] = {"metadata": metadata}

    if config.include_same_surface:
        results["same_surface"] = _evaluate_same_surface(config, model, device)

    external_results = _evaluate_external_datasets(
        config, model, device, git_sha=None, git_dirty=None, model_identity=model_identity
    )
    results.update(external_results)
    _write_launcher_summary(config.output.output_dir / "launcher_summary.json", config, results)
    return results


def _evaluate_same_surface(config: EvaluationConfig, model, device) -> dict[str, object]:
    loaders = create_supervised_dataloaders(config.data, config.split, config.runtime)
    loader = getattr(loaders, config.split_name)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    amp_enabled = config.runtime.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for inputs, batch_targets in loader:
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(move_model_input(inputs, device))
            predictions.append(outputs.cpu())
            targets.append(batch_targets.cpu())

    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    metrics = calculate_regression_metrics(prediction_tensor, target_tensor)
    results_frame = pd.DataFrame(
        {
            "Actual": target_tensor.flatten().numpy(),
            "Predicted": prediction_tensor.flatten().numpy(),
        }
    )
    results_frame["Error"] = results_frame["Predicted"] - results_frame["Actual"]
    results_frame["Absolute_Error"] = results_frame["Error"].abs()

    results_path = config.output.output_dir / config.output.evaluation_results_name
    results_frame.to_csv(results_path, index=False)
    plot_evaluation_results(
        target_tensor.flatten().numpy(),
        prediction_tensor.flatten().numpy(),
        metrics,
        config.output.output_dir / config.output.evaluation_plot_name,
        config.output.output_dir / config.output.prediction_plot_name,
    )
    write_metrics_summary(metrics, config.output.output_dir / config.output.metrics_summary_name)
    return {
        "metrics": metrics,
        "results_path": results_path,
    }


def _evaluate_external_datasets(
    config: EvaluationConfig,
    model,
    device,
    *,
    git_sha: str | None,
    git_dirty: bool | None,
    model_identity: dict[str, str],
) -> dict[str, object]:
    checkpoint_stem = Path(config.checkpoint_path).stem
    root_output_dir = config.output.output_dir / checkpoint_stem
    if root_output_dir.exists() and any(root_output_dir.iterdir()):
        raise ValueError(f"Frozen evaluation refuses a non-empty result directory: {root_output_dir}")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    pointwise_metric_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    baseline_metadata_rows: list[dict[str, object]] = []
    render_rows: list[dict[str, object]] = []
    geometry_partition_rows: list[dict[str, Any]] = []
    prediction_failures: list[dict[str, str]] = []

    if config.data.csv_path is None or config.data.image_dir is None:
        raise ValueError("External evaluation requires the data1 paths for fixed baselines.")
    duplicate_counts = assert_no_exact_cross_role_duplicates(
        config.data.image_dir,
        {dataset.name: dataset.image_dir for dataset in config.external_datasets},
    )

    for dataset in config.external_datasets:
        dataset_output_dir = resolve_standard_evaluation_output_dir(root_output_dir, dataset.name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            results_frame = _predict_dataset(
                runtime_config=config.runtime,
                image_size=config.data.image_size,
                dataset=dataset,
                model=model,
                device=device,
            )
        except Exception as exc:
            failure = {
                "dataset_name": dataset.name,
                "status": "failed",
                "reason": f"prediction_error:{exc}",
            }
            prediction_failures.append(failure)
            render_rows.append(failure)
            continue

        prediction_csv = dataset_output_dir / config.output.prediction_csv_name
        results_frame.to_csv(prediction_csv, index=False)
        stats = summarize_prediction_distribution(dataset.name, results_frame["Predicted_Value"])
        (dataset_output_dir / "prediction_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        plot_prediction_histogram(results_frame["Predicted_Value"], dataset.name, dataset_output_dir / "prediction_histogram.png")

        ground_truth = load_ground_truth_frame(dataset)
        scored_frame, metrics = score_predictions_against_ground_truth(
            dataset_name=dataset.name,
            prediction_frame=results_frame,
            ground_truth_frame=ground_truth,
        )
        absolute_error = (scored_frame["Predicted_Value"] - scored_frame["Ground_Truth"]).abs()
        metrics["tolerance"] = float(config.tolerance)
        metrics["tolerance_coverage"] = float((absolute_error <= config.tolerance).mean())
        metrics = _public_error_metrics(metrics)
        pointwise_metric_rows.append(metrics)
        intervals, _ = spatial_block_bootstrap(
            scored_frame,
            tile_size=config.bootstrap_tile_size,
            repetitions=config.bootstrap_repetitions,
            seed=config.split.seed,
            tolerance=config.tolerance,
        )
        interval_rows.append({"dataset_name": dataset.name, **intervals})

        baseline_predictions, baseline_metadata = build_baseline_predictions(
            config.data.csv_path,
            config.data.image_dir,
            results_frame["Image_Name"],
            dataset.image_dir,
        )
        comparison, _ = compare_model_to_baselines(
            scored_frame,
            baseline_predictions,
            tile_size=config.bootstrap_tile_size,
            repetitions=config.bootstrap_repetitions,
            seed=config.split.seed,
        )
        baseline_rows.extend(comparison.assign(dataset_name=dataset.name).to_dict("records"))
        baseline_metadata_rows.append({"dataset_name": dataset.name, **baseline_metadata})
        scored_frame.to_csv(dataset_output_dir / "scored_predictions.csv", index=False)
        if config.include_partitions:
            geometry_partition_rows.extend(
                _ground_truth_defined_slices(dataset.name, scored_frame, tile_size=config.bootstrap_tile_size)
            )

        if config.render_enabled:
            render_result = run_matlab_render(
                prediction_csv=prediction_csv,
                output_dir=dataset_output_dir / "render",
                title_prefix=dataset.name,
                matlab_executable=config.matlab_executable,
                timeout_seconds=config.render_timeout_seconds,
                ground_truth_csv=dataset.ground_truth_csv,
            )
        else:
            render_result = {
                "status": "skipped",
                "reason": "render_disabled",
                "prediction_csv": str(prediction_csv),
                "output_dir": str(dataset_output_dir / "render"),
                "ground_truth_csv": str(dataset.ground_truth_csv) if dataset.ground_truth_csv is not None else "",
            }
        render_result["dataset_name"] = dataset.name
        render_rows.append(render_result)

    if prediction_failures:
        raise RuntimeError(f"External probe prediction failures: {prediction_failures}")

    render_frame = pd.DataFrame(render_rows)
    per_geometry_metrics_path = root_output_dir / "per_geometry_metrics.csv"
    per_geometry_slices_path = root_output_dir / "per_geometry_slices.csv"
    baseline_comparison_path = root_output_dir / "baseline_comparison.csv"
    bootstrap_intervals_path = root_output_dir / "bootstrap_intervals.csv"
    protocol_path = root_output_dir / "evaluation_protocol.json"
    render_status_path = root_output_dir / "render_status.csv"
    render_frame.to_csv(render_status_path, index=False)

    pointwise_metrics_frame = pd.DataFrame(pointwise_metric_rows)
    interval_frame = pd.DataFrame(interval_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    pointwise_metrics_frame.to_csv(per_geometry_metrics_path, index=False)
    interval_frame.to_csv(bootstrap_intervals_path, index=False)
    baseline_frame.to_csv(baseline_comparison_path, index=False)

    pd.DataFrame(geometry_partition_rows).to_csv(per_geometry_slices_path, index=False)
    per_geometry_acceptance = []
    for dataset_name in CANONICAL_EXTERNAL_DATASETS:
        rows = baseline_frame.loc[baseline_frame["dataset_name"] == dataset_name]
        per_geometry_acceptance.append(
            {
                "dataset_name": dataset_name,
                "passes_both_preregistered_baselines": bool(
                    len(rows) == 2 and rows["model_significantly_better"].astype(bool).all()
                ),
            }
        )
    protocol = {
        "protocol_version": "depth-regression-v1",
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_sha256": file_sha256(config.checkpoint_path),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "model_id": model_identity.get("model_id"),
        "model_version": model_identity.get("version"),
        "training_csv": str(config.data.csv_path),
        "training_csv_sha256": file_sha256(config.data.csv_path),
        "external_datasets": [
            {
                "name": dataset.name,
                "ground_truth_csv": str(dataset.ground_truth_csv),
                "ground_truth_sha256": file_sha256(dataset.ground_truth_csv),
                "exact_train_image_duplicates": duplicate_counts[dataset.name],
            }
            for dataset in config.external_datasets
        ],
        "primary_metric": "mae_per_geometry",
        "bootstrap": {
            "tile_size": config.bootstrap_tile_size,
            "repetitions": config.bootstrap_repetitions,
            "seed": config.split.seed,
        },
        "tolerance_coverage_threshold": config.tolerance,
        "baseline_acceptance": "reported separately: delta_mae_ci_high < 0 for constant and weak_ridge",
        "per_geometry_acceptance": per_geometry_acceptance,
        "historical_test_contamination": True,
        "claim_scope": "locked retrospective baseline; not a pristine first-look blind test",
        "baseline_metadata": baseline_metadata_rows,
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": root_output_dir,
        "per_geometry_metrics_path": per_geometry_metrics_path,
        "per_geometry_slices_path": per_geometry_slices_path,
        "baseline_comparison_path": baseline_comparison_path,
        "bootstrap_intervals_path": bootstrap_intervals_path,
        "protocol_path": protocol_path,
        "render_status_path": render_status_path,
        "per_geometry_metrics": pointwise_metrics_frame,
        "bootstrap_intervals": interval_frame,
        "baseline_comparison": baseline_frame,
        "render_status": render_frame,
        "per_geometry_slices": geometry_partition_rows,
    }


def _public_error_metrics(metrics: dict[str, float | str]) -> dict[str, float | str]:
    names = (
        "dataset_name",
        "sample_size",
        "mae",
        "rmse",
        "median_absolute_error",
        "p95_absolute_error",
        "max_absolute_error",
        "bias",
        "tolerance",
        "tolerance_coverage",
    )
    return {name: metrics[name] for name in names}


def _predict_dataset(
    runtime_config: RuntimeConfig,
    image_size: tuple[int, int],
    dataset: StandardEvaluationDatasetSpec,
    model,
    device,
) -> pd.DataFrame:
    data_config = DataConfig(
        image_dir=dataset.image_dir,
        input_mode=dataset.input_mode,
        image_size=image_size,
        sr_image_dir=dataset.sr_image_dir,
        sr_manifest_path=dataset.sr_manifest_path,
        valid_extensions=dataset.valid_extensions,
    )
    loader = create_image_prediction_loader(data_config, runtime_config)
    filenames: list[str] = []
    predictions: list[float] = []
    amp_enabled = runtime_config.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for inputs, batch_filenames in loader:
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(move_model_input(inputs, device)).cpu().flatten().tolist()
            filenames.extend(batch_filenames)
            predictions.extend(outputs)
    return pd.DataFrame({"Image_Name": filenames, "Predicted_Value": predictions})


def _ground_truth_defined_slices(
    dataset_name: str,
    scored_frame: pd.DataFrame,
    *,
    tile_size: int,
) -> list[dict[str, Any]]:
    """Report raw errors on slices defined without consulting model predictions."""
    if not {"X", "Y", "Ground_Truth", "Predicted_Value"}.issubset(scored_frame.columns):
        raise ValueError("Ground-truth slices require X, Y, Ground_Truth and Predicted_Value.")
    frame = scored_frame.copy()
    frame["signed_error"] = frame["Predicted_Value"].astype(float) - frame["Ground_Truth"].astype(float)
    frame["tile_x"] = ((frame["X"].astype(float) - float(frame["X"].min())) // tile_size).astype(int)
    frame["tile_y"] = ((frame["Y"].astype(float) - float(frame["Y"].min())) // tile_size).astype(int)
    rows: list[dict[str, Any]] = []
    for (tile_x, tile_y), part in frame.groupby(["tile_x", "tile_y"], sort=True):
        rows.append(_partition_stats(part, dataset_name, "coordinate_tile", f"x{tile_x}_y{tile_y}"))

    unique_depths = np.sort(frame["Ground_Truth"].astype(float).unique())
    if len(unique_depths) <= 12:
        for depth, part in frame.groupby("Ground_Truth", sort=True):
            rows.append(_partition_stats(part, dataset_name, "ground_truth_depth", f"{float(depth):.12g}"))
    else:
        frame["ground_truth_depth_bin"] = pd.qcut(frame["Ground_Truth"], q=4, duplicates="drop")
        for interval, part in frame.groupby("ground_truth_depth_bin", sort=True, observed=True):
            rows.append(_partition_stats(part, dataset_name, "ground_truth_depth_bin", str(interval)))
    return rows


def _partition_stats(
    frame: pd.DataFrame,
    dataset_name: str,
    partition_kind: str,
    partition_name: str,
) -> dict[str, Any]:
    errors = frame["signed_error"].to_numpy(dtype=float)
    absolute_errors = np.abs(errors)
    return {
        "dataset_name": dataset_name,
        "partition_kind": partition_kind,
        "partition_name": partition_name,
        "sample_size": int(len(frame)),
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "bias": float(errors.mean()),
        "max_absolute_error": float(absolute_errors.max()),
        "p95_absolute_error": float(np.quantile(absolute_errors, 0.95)),
    }


def validate_standard_evaluation_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip()
    if not normalized:
        raise ValueError("Dataset name must not be empty.")
    if any(token in normalized for token in ("..", "/", "\\")):
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return normalized


def resolve_standard_evaluation_output_dir(root_output_dir: Path, dataset_name: str) -> Path:
    safe_name = validate_standard_evaluation_dataset_name(dataset_name)
    root_resolved = Path(root_output_dir).resolve()
    dataset_output_dir = (root_resolved / safe_name).resolve()
    dataset_output_dir.relative_to(root_resolved)
    return dataset_output_dir


def _write_launcher_summary(path: Path, config: EvaluationConfig, results: dict[str, object]) -> None:
    payload = {
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path is not None else "",
        "output_dir": str(config.output.output_dir),
        "dataset_name": Path(config.data.csv_path).stem if config.data.csv_path is not None else "",
        "same_surface_enabled": config.include_same_surface,
        "external_datasets": [dataset.name for dataset in config.external_datasets],
        "result": _jsonify(results),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonify(tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value

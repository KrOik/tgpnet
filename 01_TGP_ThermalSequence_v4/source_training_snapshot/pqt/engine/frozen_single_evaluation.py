from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from pqt.assets import verify_assets
from pqt.config.defaults import build_default_standard_evaluation_config
from pqt.config.schema import ExperimentConfig, ModelConfig, RuntimeConfig
from pqt.engine.evaluation.baselines import (
    assert_no_exact_cross_role_duplicates,
    build_baseline_predictions,
    compare_model_to_baselines,
)
from pqt.engine.evaluation.scoring import load_ground_truth_frame, score_predictions_against_ground_truth, spatial_block_bootstrap
from pqt.engine.runtime import resolve_device, seed_runtime
from pqt.engine.standard_evaluation import CANONICAL_EXTERNAL_DATASETS, _ground_truth_defined_slices, _predict_dataset
from pqt.experiments.loader import resolve_experiment
from pqt.model.checkpoint import load_checkpoint, load_trusted_checkpoint_payload, model_config_from_checkpoint
from pqt.model.factory import build_model


PROTOCOL = "frozen-single-checkpoint-geometry-evaluation-v1"


def _validate_resnet100_checkpoint(metadata: dict[str, Any]) -> dict[str, Any] | None:
    config = metadata.get("config")
    if not isinstance(config, dict):
        return None
    split = config.get("split", {})
    selection = config.get("best_checkpoint", {})
    model = config.get("model", {})
    optimization = config.get("optimization", {})
    runtime = config.get("runtime", {})
    observed = {
        "split_strategy": split.get("strategy"),
        "split_seed": split.get("seed"),
        "selection_strategy": selection.get("strategy"),
        "backbone": model.get("backbone_name"),
        "head": model.get("head_variant"),
        "loss": optimization.get("loss_name"),
        "huber_delta": optimization.get("huber_delta"),
        "epochs": optimization.get("epochs"),
        "optimization_seed": runtime.get("optimization_seed"),
    }
    expected = {
        "split_strategy": "full",
        "split_seed": 42,
        "selection_strategy": "fixed_epochs",
        "backbone": "resnet50",
        "head": "canonical",
        "loss": "huber",
        "huber_delta": 0.16,
        "epochs": 100,
        "optimization_seed": 99173,
    }
    if observed != expected:
        raise ValueError(f"Frozen ResNet100 checkpoint protocol mismatch: {observed}")
    if int(metadata.get("epoch", -1)) != 100:
        raise ValueError("Frozen ResNet100 checkpoint must be saved at epoch 100.")
    return config


def run_frozen_single_geometry_evaluation(
    freeze_spec_path: Path,
    output_dir: Path,
    *,
    code_root: Path,
    asset_root: Path,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    code_root = Path(code_root).resolve()
    asset_root = Path(asset_root).resolve()
    freeze_spec_path = Path(freeze_spec_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Frozen evaluation refuses non-empty output directory: {output_dir}")

    spec = json.loads(freeze_spec_path.read_text(encoding="utf-8"))

    errors = verify_assets(asset_root, code_root)
    if errors:
        raise ValueError("Frozen asset verification failed: " + "; ".join(errors))
    checkpoint_path = Path(spec["checkpoint_path"]).resolve()
    payload = load_trusted_checkpoint_payload(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Unsupported frozen checkpoint payload.")
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    _validate_resnet100_checkpoint(metadata)

    evaluation = build_default_standard_evaluation_config(asset_root=asset_root, output_dir=output_dir)
    names = tuple(dataset.name for dataset in evaluation.external_datasets)
    if names != CANONICAL_EXTERNAL_DATASETS:
        raise ValueError(f"Expected exactly {CANONICAL_EXTERNAL_DATASETS}, got {names}.")
    duplicate_counts = assert_no_exact_cross_role_duplicates(
        evaluation.data.image_dir,
        {dataset.name: dataset.image_dir for dataset in evaluation.external_datasets},
    )
    evaluation.runtime = runtime
    seed_runtime(42, runtime.deterministic)
    device = resolve_device(runtime)
    experiment = resolve_experiment(
        ExperimentConfig(
            experiment_id=str(spec["experiment_id"]), root_dir=code_root / "experiments", validate_branch=False
        )
    )
    model_config = model_config_from_checkpoint(
        metadata, fallback=ModelConfig(backbone_name="resnet50", head_variant="canonical")
    )
    model = build_model(model_config, experiment=experiment).to(device)
    strict_metadata = load_checkpoint(checkpoint_path, model=model, map_location=device, strict=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    baseline_sample_rows: list[dict[str, object]] = []
    for dataset in evaluation.external_datasets:
        predictions = _predict_dataset(runtime, evaluation.data.image_size, dataset, model, device)
        ground_truth = load_ground_truth_frame(dataset)
        scored, metrics = score_predictions_against_ground_truth(dataset.name, predictions, ground_truth)
        absolute_error = (scored["Predicted_Value"] - scored["Ground_Truth"]).abs()
        metric_rows.append(
            {
                **metrics,
                "sample_size": int(len(scored)),
                "tolerance": float(evaluation.tolerance),
                "tolerance_coverage": float((absolute_error <= evaluation.tolerance).mean()),
            }
        )
        intervals, _ = spatial_block_bootstrap(
            scored,
            tile_size=evaluation.bootstrap_tile_size,
            repetitions=evaluation.bootstrap_repetitions,
            seed=42,
            tolerance=evaluation.tolerance,
        )
        interval_rows.append({"dataset_name": dataset.name, **intervals})
        slice_rows.extend(_ground_truth_defined_slices(dataset.name, scored, tile_size=evaluation.bootstrap_tile_size))
        baselines, _ = build_baseline_predictions(
            evaluation.data.csv_path, evaluation.data.image_dir, predictions["Image_Name"], dataset.image_dir
        )
        comparisons, samples = compare_model_to_baselines(
            scored,
            baselines,
            tile_size=evaluation.bootstrap_tile_size,
            repetitions=evaluation.bootstrap_repetitions,
            seed=42,
        )
        baseline_rows.extend(comparisons.assign(dataset_name=dataset.name).to_dict("records"))
        baseline_sample_rows.extend(samples.assign(dataset_name=dataset.name).to_dict("records"))
        dataset_dir = output_dir / "predictions" / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        scored.to_csv(dataset_dir / "scored_predictions.csv", index=False)

    metrics_path = output_dir / "per_geometry_metrics.csv"
    intervals_path = output_dir / "bootstrap_intervals.csv"
    slices_path = output_dir / "per_geometry_slices.csv"
    baselines_path = output_dir / "baseline_comparison.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(interval_rows).to_csv(intervals_path, index=False)
    pd.DataFrame(slice_rows).to_csv(slices_path, index=False)
    pd.DataFrame(baseline_rows).to_csv(baselines_path, index=False)
    pd.DataFrame(baseline_sample_rows).to_csv(output_dir / "baseline_bootstrap_samples.csv", index=False)
    protocol = {
        "protocol": PROTOCOL,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "freeze_spec_path": str(freeze_spec_path),
        "freeze_spec_sha256": file_sha256(freeze_spec_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": strict_metadata.get("epoch"),
        "model_config": asdict(model_config),
        "external_datasets": list(names),
        "exact_train_image_duplicates": duplicate_counts,
        "selection_uses_external_data": False,
        "claim_scope": "frozen post-training external validation; no checkpoint or hyperparameter selection",
    }
    protocol_path = output_dir / "evaluation_protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": output_dir,
        "per_geometry_metrics_path": metrics_path,
        "bootstrap_intervals_path": intervals_path,
        "per_geometry_slices_path": slices_path,
        "baseline_comparison_path": baselines_path,
        "protocol_path": protocol_path,
    }

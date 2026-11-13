from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pqt.assets import verify_assets
from pqt.config.defaults import build_default_standard_evaluation_config
from pqt.config.schema import ExperimentConfig, ModelConfig, RuntimeConfig
from pqt.engine.candidate_selection import promotion_decision
from pqt.engine.evaluation.baselines import build_baseline_predictions
from pqt.engine.evaluation.scoring import (
    load_ground_truth_frame,
    paired_model_block_bootstrap,
    score_predictions_against_ground_truth,
    spatial_block_bootstrap,
)
from pqt.engine.runtime import resolve_device, seed_runtime
from pqt.engine.standard_evaluation import _partition_stats, _predict_dataset
from pqt.experiments.loader import resolve_experiment
from pqt.model.checkpoint import load_checkpoint, load_trusted_checkpoint_payload, model_config_from_checkpoint
from pqt.model.factory import build_model


def _fallback_model_config(name: str) -> ModelConfig:
    """Return the declared architecture for legacy checkpoints without config metadata."""
    fallback_by_name = {
        "v0.3.0": ModelConfig(backbone_name="convnext_tiny", head_variant="canonical"),
        "efficientnet_v2_s": ModelConfig(backbone_name="efficientnet_v2_s", head_variant="canonical"),
        "resnet50": ModelConfig(backbone_name="resnet50", head_variant="canonical"),
    }
    return fallback_by_name.get(name, ModelConfig())


def _validate_checkpoint_protocol(metadata: dict[str, Any], name: str) -> None:
    config = metadata.get("config")
    if not isinstance(config, dict):
        return
    split = config.get("split", {})
    selection = config.get("best_checkpoint", {})
    if split.get("strategy") != "full" or selection.get("strategy") not in {"fixed_epochs", "train_loss"} or split.get("seed") != 42:
        raise ValueError(f"{name} is not a seed-42 full-data checkpoint selected without external data.")
    data = config.get("data", {})
    if Path(data.get("csv_path", "")).name != "data1.csv":
        raise ValueError(f"{name} was not trained on canonical data1.")


def _quartile_slices(dataset_name: str, scored: pd.DataFrame) -> list[dict[str, object]]:
    frame = scored.copy()
    frame["quartile"] = pd.qcut(frame["Ground_Truth"], q=4, duplicates="drop")
    rows = []
    for index, (interval, part) in enumerate(frame.groupby("quartile", sort=True, observed=True), start=1):
        row = _partition_stats(part, dataset_name, "ground_truth_quartile", f"Q{index}:{interval}")
        rows.append(row)
    return rows


def run_frozen_candidate_comparison(
    freeze_spec_path: Path,
    output_dir: Path,
    *,
    code_root: Path,
    asset_root: Path,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    """Evaluate a declared candidate set without changing release-evaluator policy."""
    code_root = Path(code_root).resolve()
    asset_root = Path(asset_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Candidate comparison refuses non-empty output directory: {output_dir}")
    spec = json.loads(Path(freeze_spec_path).read_text(encoding="utf-8"))
    models = spec.get("models")
    if not isinstance(models, list) or len(models) < 4:
        raise ValueError("Freeze spec must contain all candidate and reference models.")
    names = [str(item.get("name")) for item in models]
    if len(set(names)) != len(names) or not {"candidate", "v0.3.0", "efficientnet_v2_s", "resnet50"}.issubset(names):
        raise ValueError("Freeze spec has missing or duplicate required model names.")
    errors = verify_assets(asset_root, code_root)
    if errors:
        raise ValueError("Frozen asset verification failed: " + "; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "freeze_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    evaluation = build_default_standard_evaluation_config(asset_root=asset_root, output_dir=output_dir)
    evaluation.runtime = runtime
    seed_runtime(42, runtime.deterministic)
    device = resolve_device(runtime)
    experiment_id = spec.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("Freeze spec must identify the experiment implementation.")
    experiment = resolve_experiment(ExperimentConfig(experiment_id=experiment_id,
                                                       root_dir=code_root / "experiments", validate_branch=False))

    loaded_models: dict[str, torch.nn.Module] = {}
    identities: list[dict[str, object]] = []
    for item in models:
        name = str(item["name"])
        checkpoint_path = Path(item["checkpoint_path"]).resolve()
        payload = load_trusted_checkpoint_payload(checkpoint_path, map_location="cpu")
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
        _validate_checkpoint_protocol(metadata, name)
        declared_config = item.get("model_config")
        fallback = ModelConfig(**declared_config) if isinstance(declared_config, dict) else _fallback_model_config(name)
        model_config = model_config_from_checkpoint(metadata, fallback=fallback)
        model = build_model(model_config, experiment=experiment).to(device)
        strict_metadata = load_checkpoint(checkpoint_path, model=model, map_location=device, strict=True)
        model.eval()
        loaded_models[name] = model
        identities.append({"name": name, "checkpoint_path": str(checkpoint_path),
                           "epoch": strict_metadata.get("epoch"), "model_config": asdict(model_config)})

    metric_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    paired_sample_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for dataset in evaluation.external_datasets:
        ground_truth = load_ground_truth_frame(dataset)
        scored: dict[str, pd.DataFrame] = {}
        prediction_names = None
        for name, model in loaded_models.items():
            predictions = _predict_dataset(runtime, evaluation.data.image_size, dataset, model, device)
            prediction_names = predictions["Image_Name"]
            model_scored, metrics = score_predictions_against_ground_truth(dataset.name, predictions, ground_truth)
            scored[name] = model_scored
            metric_rows.append({"model_name": name, **metrics})
            intervals, _ = spatial_block_bootstrap(
                model_scored, tile_size=evaluation.bootstrap_tile_size,
                repetitions=evaluation.bootstrap_repetitions, seed=42,
            )
            bootstrap_rows.append({"model_name": name, "dataset_name": dataset.name, **intervals})
            for row in _quartile_slices(dataset.name, model_scored):
                slice_rows.append({"model_name": name, **row})
            model_dir = output_dir / "predictions" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_scored.to_csv(model_dir / f"{dataset.name}.csv", index=False)
        assert prediction_names is not None
        baselines, _ = build_baseline_predictions(evaluation.data.csv_path, evaluation.data.image_dir,
                                                  prediction_names, dataset.image_dir)
        for baseline_name, column in (("constant", "constant_prediction"), ("weak_ridge", "weak_ridge_prediction")):
            predictions = baselines[["Image_Name", column]].rename(columns={column: "Predicted_Value"})
            baseline_scored, metrics = score_predictions_against_ground_truth(dataset.name, predictions, ground_truth)
            scored[baseline_name] = baseline_scored
            metric_rows.append({"model_name": baseline_name, **metrics})
            intervals, _ = spatial_block_bootstrap(
                baseline_scored, tile_size=evaluation.bootstrap_tile_size,
                repetitions=evaluation.bootstrap_repetitions, seed=42,
            )
            bootstrap_rows.append({"model_name": baseline_name, "dataset_name": dataset.name, **intervals})
            for row in _quartile_slices(dataset.name, baseline_scored):
                slice_rows.append({"model_name": baseline_name, **row})
        summaries, samples = paired_model_block_bootstrap(
            scored, candidate_name="candidate", tile_size=evaluation.bootstrap_tile_size,
            repetitions=evaluation.bootstrap_repetitions, seed=42,
        )
        paired_rows.extend(summaries.assign(dataset_name=dataset.name).to_dict("records"))
        paired_sample_rows.extend(samples.assign(dataset_name=dataset.name).to_dict("records"))

    metrics_frame = pd.DataFrame(metric_rows)
    paired_frame = pd.DataFrame(paired_rows)
    decision = promotion_decision(metrics_frame.to_dict("records"), paired_frame.to_dict("records"))
    metrics_frame.to_csv(output_dir / "per_geometry_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(output_dir / "per_geometry_slices.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "bootstrap_intervals.csv", index=False)
    paired_frame.to_csv(output_dir / "paired_intervals.csv", index=False)
    pd.DataFrame(paired_sample_rows).to_csv(output_dir / "paired_bootstrap_samples.csv", index=False)
    protocol = {
        "protocol": "candidate-comparison-v1",
        "bootstrap": {"tile_size": evaluation.bootstrap_tile_size,
                      "repetitions": evaluation.bootstrap_repetitions, "seed": 42},
        "models": identities, "promotion": decision,
    }
    (output_dir / "comparison_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
    return {"output_dir": output_dir, "promotion": decision, "models": identities}

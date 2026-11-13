from __future__ import annotations

import csv
import json
import os
import secrets
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pqt.assets import verify_assets
from pqt.config.defaults import (
    build_default_prediction_config,
    build_default_standard_evaluation_config,
    build_default_train_config,
)
from pqt.config.schema import EvaluationConfig, PredictionConfig, TrainConfig
from pqt.experiments.contracts import ResolvedExperiment
from pqt.experiments.loader import apply_experiment_train_config, resolve_experiment
from pqt.workflows.evaluate import run_evaluation
from pqt.workflows.predict import run_prediction
from pqt.workflows.train import run_training
from pqt.workflows.diagnose import run_model_diagnostics
from pqt.workspace import WorkspaceLayout, repository_root


RunMode = Literal["train", "evaluate", "predict", "diagnose", "verify"]


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    code_root: Path
    asset_root: Path
    run_root: Path

    @classmethod
    def discover(cls, code_root: Path | None = None) -> "WorkspacePaths":
        code = repository_root(code_root)
        layout = WorkspaceLayout.discover(code)
        return cls(
            code_root=code,
            asset_root=layout.primary_root,
            run_root=layout.runs_root / layout.checkout_name(code),
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    mode: RunMode
    dataset_name: str = "data1"
    experiment_id: str | None = None
    checkpoint_path: Path | None = None
    image_dir: Path | None = None
    run_id: str | None = None
    runtime_overrides: dict[str, Any] | None = None
    split_strategy: str | None = None
    cv_fold: int | None = None
    epochs: int | None = None
    shuffle_train_labels: bool = False
    model_overrides: dict[str, Any] | None = None
    data_overrides: dict[str, Any] | None = None
    optimization_overrides: dict[str, Any] | None = None
    resume_state_path: Path | None = None
    segment_end_epoch: int | None = None


@dataclass(slots=True)
class ResolvedRun:
    spec: RunSpec
    paths: WorkspacePaths
    run_id: str
    run_dir: Path
    experiment: ResolvedExperiment
    config: TrainConfig | EvaluationConfig | PredictionConfig | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generated_run_id(dataset_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{dataset_name}-{stamp}-{secrets.token_hex(3)}"


def _canonical_model_row(asset_root: Path) -> dict[str, str]:
    active_path = asset_root / "manifests" / "active_experiment.json"
    if active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if not isinstance(active, dict) or not active.get("experiment_id"):
            raise RuntimeError(f"Invalid active experiment manifest: {active_path}")
        checkpoint = active.get("checkpoint")
        return {
            "model_id": str(active["experiment_id"]),
            "version": str(active.get("version", "")),
            "relative_path": str(checkpoint) if checkpoint else "",
        }
    with (asset_root / "manifests" / "models.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    canonical = [row for row in rows if row.get("role", "canonical") == "canonical"]
    if len(canonical) != 1:
        raise RuntimeError(f"Expected exactly one canonical model manifest row, got {len(canonical)}.")
    return canonical[0]


def _apply_runtime_overrides(config: Any, overrides: dict[str, Any] | None) -> None:
    for name, value in (overrides or {}).items():
        if not hasattr(config.runtime, name):
            raise ValueError(f"Unknown runtime override {name!r}.")
        setattr(config.runtime, name, value)


def _apply_section_overrides(section: Any, overrides: dict[str, Any] | None, section_name: str) -> None:
    for name, value in (overrides or {}).items():
        if not hasattr(section, name):
            raise ValueError(f"Unknown {section_name} override {name!r}.")
        setattr(section, name, value)


def resolve_run(spec: RunSpec, paths: WorkspacePaths | None = None) -> ResolvedRun:
    paths = paths or WorkspacePaths.discover()
    model_row = _canonical_model_row(paths.asset_root)
    experiment_id = spec.experiment_id or model_row["model_id"]
    run_id = spec.run_id or _generated_run_id(spec.dataset_name)
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in run_id):
        raise ValueError("run_id must contain only letters, digits, '-', '_' or '.'.")
    run_dir = paths.run_root / spec.mode / run_id

    train_config = build_default_train_config(
        dataset_name=spec.dataset_name,
        asset_root=paths.asset_root,
        output_dir=run_dir,
    )
    train_config.experiment.experiment_id = experiment_id
    train_config.experiment.root_dir = paths.code_root / "experiments"
    train_config.experiment.validate_branch = False
    experiment = resolve_experiment(train_config.experiment)
    if experiment is None:
        raise RuntimeError("The canonical harness requires an experiment.")
    active_train = apply_experiment_train_config(train_config, experiment)
    _apply_section_overrides(active_train.model, spec.model_overrides, "model")
    _apply_section_overrides(active_train.data, spec.data_overrides, "data")
    _apply_section_overrides(active_train.optimization, spec.optimization_overrides, "optimization")
    active_train.model.__post_init__()
    active_train.optimization.__post_init__()
    if spec.split_strategy is not None:
        active_train.split.strategy = spec.split_strategy
        active_train.best_checkpoint.strategy = "train_loss" if spec.split_strategy == "full" else "val_mae"
    if spec.cv_fold is not None:
        active_train.split.fold = spec.cv_fold
    if spec.epochs is not None:
        if spec.epochs < 1:
            raise ValueError("epochs must be positive.")
        active_train.optimization.epochs = spec.epochs
    if spec.resume_state_path is not None:
        active_train.optimization.resume_state_path = Path(spec.resume_state_path)
    if spec.segment_end_epoch is not None:
        active_train.optimization.segment_end_epoch = spec.segment_end_epoch
    active_train.optimization.__post_init__()
    active_train.data.shuffle_train_labels = spec.shuffle_train_labels
    active_train.data.__post_init__()

    config: TrainConfig | EvaluationConfig | PredictionConfig | None
    if spec.mode == "train":
        config = active_train
        config.output.output_dir = run_dir
    elif spec.mode == "diagnose":
        if spec.checkpoint_path is None and not model_row["relative_path"]:
            raise FileNotFoundError("No frozen v1 checkpoint is registered; diagnose requires --checkpoint-path.")
        config = active_train
        config.output.output_dir = run_dir
        config.diagnostics.enabled = True
        config.diagnostics.run_ablations = True
        config.diagnostics.checkpoint_path = spec.checkpoint_path or (paths.asset_root / model_row["relative_path"])
    elif spec.mode == "evaluate":
        if spec.checkpoint_path is None and not model_row["relative_path"]:
            raise FileNotFoundError("No frozen v1 checkpoint is registered; evaluate requires --checkpoint-path.")
        checkpoint = spec.checkpoint_path or (paths.asset_root / model_row["relative_path"])
        config = build_default_standard_evaluation_config(
            checkpoint_path=checkpoint,
            output_dir=run_dir,
            dataset_name=spec.dataset_name,
            asset_root=paths.asset_root,
        )
        config.model = active_train.model
        config.data.image_size = active_train.data.image_size
        config.runtime = deepcopy(active_train.runtime)
        config.experiment = active_train.experiment
    elif spec.mode == "predict":
        if spec.checkpoint_path is None and not model_row["relative_path"]:
            raise FileNotFoundError("No frozen v1 checkpoint is registered; predict requires --checkpoint-path.")
        checkpoint = spec.checkpoint_path or (paths.asset_root / model_row["relative_path"])
        config = build_default_prediction_config(
            image_dir=spec.image_dir,
            checkpoint_path=checkpoint,
            asset_root=paths.asset_root,
            output_dir=run_dir,
        )
        config.model = active_train.model
        config.runtime = deepcopy(active_train.runtime)
        config.experiment = active_train.experiment
    else:
        config = None
    if config is not None:
        _apply_runtime_overrides(config, spec.runtime_overrides)
        config.runtime.__post_init__()
    return ResolvedRun(spec, paths, run_id, run_dir, experiment, config)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape)}
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _git_value(paths: WorkspacePaths, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=paths.code_root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _source_git_identity(paths: WorkspacePaths) -> tuple[str, bool]:
    source_sha = os.environ.get("TGPNET_SOURCE_GIT_SHA")
    source_dirty = os.environ.get("TGPNET_SOURCE_GIT_DIRTY")
    if source_sha is None and source_dirty is None:
        return (
            _git_value(paths, "rev-parse", "HEAD"),
            bool(_git_value(paths, "status", "--porcelain")),
        )
    if source_sha is None or source_dirty is None:
        raise ValueError("TGPNET_SOURCE_GIT_SHA and TGPNET_SOURCE_GIT_DIRTY must be provided together.")
    if len(source_sha) != 40 or any(character not in "0123456789abcdef" for character in source_sha.lower()):
        raise ValueError("TGPNET_SOURCE_GIT_SHA must be a full 40-character hexadecimal commit.")
    if source_dirty not in {"0", "1"}:
        raise ValueError("TGPNET_SOURCE_GIT_DIRTY must be '0' or '1'.")
    return source_sha.lower(), source_dirty == "1"


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_run(resolved: ResolvedRun) -> dict[str, Any]:
    resolved.paths.run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved.run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resuming = bool(
        isinstance(resolved.config, TrainConfig)
        and resolved.config.optimization.resume_state_path is not None
    )
    if resolved.run_dir.exists() and not resuming:
        raise FileExistsError(resolved.run_dir)
    resolved.run_dir.mkdir(mode=0o700, exist_ok=resuming)
    manifest_path = resolved.run_dir / "run.json"
    git_sha, git_dirty = _source_git_identity(resolved.paths)
    payload: dict[str, Any] = {
        "run_id": resolved.run_id,
        "mode": resolved.spec.mode,
        "status": "running",
        "started_at": _utc_now(),
        "code_root": resolved.paths.code_root,
        "asset_root": resolved.paths.asset_root,
        "run_dir": resolved.run_dir,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "experiment_id": resolved.experiment.manifest.experiment_id,
        "experiment_version": resolved.experiment.manifest.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "spec": asdict(resolved.spec),
        "config": asdict(resolved.config) if resolved.config is not None else None,
    }
    if isinstance(resolved.config, TrainConfig):
        optimization_seed = resolved.config.runtime.optimization_seed
        payload["seed_identity"] = {
            "split_seed": resolved.config.split.seed,
            "optimization_seed": optimization_seed,
            "effective_optimization_seed": resolved.config.runtime.effective_optimization_seed(
                resolved.config.split.seed
            ),
        }
    if resuming and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("run_id") != resolved.run_id or prior.get("experiment_id") != resolved.experiment.manifest.experiment_id:
            raise ValueError("Resume state does not belong to this run identity.")
        payload["segments"] = list(prior.get("segments", []))
    _write_manifest(manifest_path, payload)
    try:
        if resolved.spec.mode == "train":
            result = run_training(
                config=resolved.config,
                experiment=resolved.experiment,
                config_is_resolved=True,
            )
        elif resolved.spec.mode == "evaluate":
            result = run_evaluation(config=resolved.config, experiment=resolved.experiment)
        elif resolved.spec.mode == "predict":
            result = run_prediction(config=resolved.config, experiment=resolved.experiment)
        elif resolved.spec.mode == "diagnose":
            result = run_model_diagnostics(config=resolved.config, experiment=resolved.experiment)
        else:
            errors = verify_assets(resolved.paths.asset_root, resolved.paths.code_root)
            if errors:
                raise RuntimeError("; ".join(errors))
            result = {"verified": True}
        training_complete = bool(result.get("training_complete", True)) if isinstance(result, dict) else True
        payload["status"] = "completed" if training_complete else "segment_completed"
        payload["result"] = _jsonify(result)
        if isinstance(result, dict) and "observability_profile" in result:
            payload["observability"] = {
                "profile": result.get("observability_profile"),
                "artifacts": result.get("observability_artifacts"),
                "diagnostic_status": result.get("diagnostic_status"),
            }
        if isinstance(result, dict):
            payload.setdefault("segments", []).append(
                {
                    "completed_epochs": result.get("completed_epochs"),
                    "training_complete": training_complete,
                    "resume_state_path": result.get("resume_state_path"),
                }
            )
        return result
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["finished_at"] = _utc_now()
        _write_manifest(manifest_path, payload)

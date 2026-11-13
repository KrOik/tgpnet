from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any

import yaml

from pqt.config.schema import ExperimentConfig, ModelConfig, TrainConfig
from pqt.engine.losses import CombinedLoss, HuberRegressionLoss

from .contracts import (
    ALLOWED_EXPERIMENT_PARTS,
    ALLOWED_STATUSES,
    PART_ALIASES,
    AssemblyOverrides,
    ExperimentManifest,
    ExperimentRuntime,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root_dir(root_dir: Path) -> Path:
    return root_dir if root_dir.is_absolute() else (_repo_root() / root_dir)


def _normalize_part(part: str) -> str:
    normalized = PART_ALIASES.get(part)
    if normalized is None:
        raise ValueError(f"Unsupported experiment changed_part {part!r}. Allowed values: {sorted(ALLOWED_EXPERIMENT_PARTS)}")
    return normalized


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment manifest must be a mapping: {path}")
    return payload


def _current_git_branch(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    branch = result.stdout.strip()
    return branch or None


def _assert_within_experiment_dir(experiment_dir: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(experiment_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Experiment paths must stay within {experiment_dir}") from exc
    return resolved


def _resolve_declared_path(experiment_dir: Path, relative_path: str) -> Path:
    candidate = _assert_within_experiment_dir(experiment_dir, experiment_dir / relative_path)
    if not candidate.exists():
        raise ValueError(f"Declared experiment path does not exist: {candidate}")
    return candidate


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_callable(module: ModuleType, export_name: str, path: Path):
    value = getattr(module, export_name, None)
    if value is None or not callable(value):
        raise ValueError(f"Expected callable {export_name!r} in {path}")
    return value


def _build_manifest(experiment_dir: Path, raw: dict[str, Any], validate_branch: bool) -> ExperimentManifest:
    experiment_id = raw.get("experiment_id")
    version = raw.get("version")
    status = raw.get("status")
    branch = raw.get("branch")
    baseline = raw.get("baseline")
    parent_experiment_id = raw.get("extends")
    changed_parts_raw = raw.get("changed_parts")
    entry = raw.get("entry") or {}

    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id is required and must be a non-empty string.")
    if experiment_id != experiment_dir.name:
        raise ValueError(f"experiment_id {experiment_id!r} must match directory name {experiment_dir.name!r}.")
    if not isinstance(version, str) or not version:
        raise ValueError("version is required and must be a non-empty string.")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status must be one of {sorted(ALLOWED_STATUSES)}, got {status!r}.")
    if not isinstance(branch, str) or not branch:
        raise ValueError("branch is required and must be a non-empty string.")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("baseline is required and must be a non-empty string.")
    if parent_experiment_id is not None and (not isinstance(parent_experiment_id, str) or not parent_experiment_id):
        raise ValueError("extends must be a non-empty experiment id when provided.")
    if not isinstance(changed_parts_raw, list) or not changed_parts_raw:
        raise ValueError("changed_parts must be a non-empty list.")

    changed_parts = tuple(_normalize_part(str(part)) for part in changed_parts_raw)
    override_decl = entry.get("overrides") if isinstance(entry, dict) else None
    if override_decl is not None and not isinstance(override_decl, dict):
        raise ValueError("entry.overrides must be a mapping when provided.")

    override_paths: dict[str, Path] = {}
    for part, default_filename in ALLOWED_EXPERIMENT_PARTS.items():
        if part == "config":
            continue
        if override_decl and part in override_decl:
            override_paths[part] = _resolve_declared_path(experiment_dir, str(override_decl[part]))
            continue
        default_path = experiment_dir / default_filename
        if default_path.exists():
            override_paths[part] = default_path.resolve()

    config_path: Path | None = None
    if isinstance(entry, dict) and "config" in entry:
        config_path = _resolve_declared_path(experiment_dir, str(entry["config"]))
    else:
        default_config = experiment_dir / ALLOWED_EXPERIMENT_PARTS["config"]
        if default_config.exists():
            config_path = default_config.resolve()

    actual_parts = set(override_paths)
    if config_path is not None:
        actual_parts.add("config")
    if set(changed_parts) != actual_parts:
        raise ValueError(
            f"changed_parts {sorted(set(changed_parts))} must match actual override files {sorted(actual_parts)} for {experiment_id}."
        )

    if validate_branch:
        current_branch = _current_git_branch(_repo_root())
        managed_main_worktree = (
            branch == "main"
            and current_branch is not None
            and current_branch in {"worktree/treeA", "worktree/treeB", "worktree/treeC", "worktree/treeD"}
        )
        if current_branch is not None and current_branch != branch and not managed_main_worktree:
            raise ValueError(f"Experiment branch {branch!r} does not match current branch {current_branch!r}.")

    return ExperimentManifest(
        experiment_id=experiment_id,
        version=version,
        status=status,
        branch=branch,
        baseline=baseline,
        changed_parts=changed_parts,
        config_path=config_path,
        override_paths=override_paths,
        parent_experiment_id=parent_experiment_id,
        notes=raw.get("notes"),
    )


def _load_local_runtime(experiment_config: ExperimentConfig) -> ExperimentRuntime | None:
    if experiment_config.experiment_id is None:
        return None

    experiments_root = _resolve_root_dir(experiment_config.root_dir)
    experiment_dir = experiments_root / experiment_config.experiment_id
    manifest_path = experiment_dir / "experiment.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")

    manifest = _build_manifest(experiment_dir, _load_yaml(manifest_path), experiment_config.validate_branch)
    runtime = ExperimentRuntime(manifest=manifest, experiment_dir=experiment_dir)

    unique_prefix = f"pqt_experiment_{manifest.experiment_id.replace('-', '_')}"
    if manifest.config_path is not None:
        config_module = _load_module(f"{unique_prefix}_config", manifest.config_path)
        runtime.config_applier = _load_callable(config_module, "apply_train_config", manifest.config_path)

    for part, path in manifest.override_paths.items():
        module = _load_module(f"{unique_prefix}_{part}", path)
        if part == "backbone":
            runtime.assembly_overrides.backbone_builder = _load_callable(module, "build_backbone", path)
        elif part == "neck":
            runtime.assembly_overrides.neck_builder = _load_callable(module, "build_neck", path)
        elif part == "head":
            runtime.assembly_overrides.head_builder = _load_callable(module, "build_head", path)
        elif part == "calibrator":
            runtime.assembly_overrides.calibrator_builder = _load_callable(module, "build_calibrator", path)
        elif part == "loss":
            runtime.loss_builder = _load_callable(module, "build_loss", path)

    return runtime


def _compose_runtime(parent: ExperimentRuntime, child: ExperimentRuntime) -> ExperimentRuntime:
    """Inherit untouched assembly parts while keeping the child identity."""
    child_overrides = child.assembly_overrides
    parent_overrides = parent.assembly_overrides
    for field_name in ("backbone_builder", "neck_builder", "head_builder", "calibrator_builder"):
        if getattr(child_overrides, field_name) is None:
            setattr(child_overrides, field_name, getattr(parent_overrides, field_name))
    if child.loss_builder is None:
        child.loss_builder = parent.loss_builder

    parent_applier = parent.config_applier
    child_applier = child.config_applier

    def apply_config(config: TrainConfig) -> TrainConfig:
        result = deepcopy(config)
        if parent_applier is not None:
            updated = parent_applier(result)
            result = result if updated is None else updated
        if child_applier is not None:
            updated = child_applier(result)
            result = result if updated is None else updated
        return result

    child.config_applier = apply_config if parent_applier is not None or child_applier is not None else None
    return child


def resolve_experiment(experiment_config: ExperimentConfig, _seen: tuple[str, ...] = ()) -> ExperimentRuntime | None:
    """Resolve one experiment and its optional ``extends`` parent chain."""
    if experiment_config.experiment_id is None:
        return None
    if experiment_config.experiment_id in _seen:
        chain = " -> ".join((*_seen, experiment_config.experiment_id))
        raise ValueError(f"Experiment inheritance cycle detected: {chain}")
    local = _load_local_runtime(experiment_config)
    if local is None or local.manifest.parent_experiment_id is None:
        return local
    parent_config = ExperimentConfig(
        experiment_id=local.manifest.parent_experiment_id,
        root_dir=experiment_config.root_dir,
        validate_branch=experiment_config.validate_branch,
    )
    parent = resolve_experiment(parent_config, (*_seen, experiment_config.experiment_id))
    if parent is None:
        raise RuntimeError("Experiment inheritance parent resolved to None.")
    return _compose_runtime(parent, local)


def apply_experiment_train_config(
    config: TrainConfig,
    runtime: ExperimentRuntime | None,
) -> TrainConfig:
    cloned = deepcopy(config)
    if runtime is None or runtime.config_applier is None:
        return cloned
    updated = runtime.config_applier(cloned)
    return cloned if updated is None else updated


def apply_experiment_model_config(
    model_config: ModelConfig,
    runtime: ExperimentRuntime | None,
) -> ModelConfig:
    """Apply the experiment's model-related defaults without leaking run paths."""
    if runtime is None or runtime.config_applier is None:
        return deepcopy(model_config)
    probe = TrainConfig(model=deepcopy(model_config))
    updated = runtime.config_applier(probe)
    return (probe if updated is None else updated).model


def build_training_loss(
    config: TrainConfig,
    runtime: ExperimentRuntime | None,
):
    if runtime is not None and runtime.loss_builder is not None:
        return runtime.loss_builder(config)
    if config.optimization.loss_name == "huber":
        return HuberRegressionLoss(delta=config.optimization.huber_delta)
    return CombinedLoss(
        mse_weight=config.optimization.mse_weight,
        l1_weight=config.optimization.l1_weight,
    )

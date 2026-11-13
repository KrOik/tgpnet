from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pqt.config.schema import ModelConfig


def _remap_legacy_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("efficientnet.features.") or key.startswith("efficientnet.avgpool."):
            remapped["backbone." + key.removeprefix("efficientnet.")] = value
            continue
        if key.startswith("efficientnet.classifier."):
            remapped["head." + key.removeprefix("efficientnet.classifier.")] = value
            continue
        if key.startswith("se_block."):
            remapped["neck.se_block." + key.removeprefix("se_block.")] = value
            continue
        if key == "temperature":
            remapped["calibrator.temperature"] = value
            continue
        remapped[key] = value
    return remapped


def unwrap_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return unwrap_model_state_dict(payload["model_state_dict"])
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return _remap_legacy_state_dict_keys(payload)


def load_trusted_checkpoint_payload(
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    # Our own checkpoints may embed Path objects in config metadata, which
    # PyTorch 2.6 rejects under the new weights_only=True default.
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: Any | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if criterion is not None:
        payload["criterion_state_dict"] = criterion.state_dict()
    if config is not None:
        payload["config"] = asdict(config) if is_dataclass(config) else config
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    checkpoint = load_trusted_checkpoint_payload(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = unwrap_model_state_dict(checkpoint)
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    else:
        state_dict = unwrap_model_state_dict(checkpoint)
        metadata = {}

    if model is not None:
        incompatible = model.load_state_dict(state_dict, strict=strict)
        metadata["missing_keys"] = list(incompatible.missing_keys)
        metadata["unexpected_keys"] = list(incompatible.unexpected_keys)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if criterion is not None:
        if isinstance(checkpoint, dict) and "criterion_state_dict" in checkpoint:
            incompatible = criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=strict)
            metadata["criterion_missing_keys"] = list(incompatible.missing_keys)
            metadata["criterion_unexpected_keys"] = list(incompatible.unexpected_keys)
            metadata["criterion_state"] = "restored"
        else:
            metadata["criterion_state"] = "missing_legacy"
    return metadata


def model_config_from_checkpoint(
    metadata: dict[str, Any], *, fallback: ModelConfig | None = None
) -> ModelConfig:
    config = metadata.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        if fallback is None:
            raise ValueError("Checkpoint does not contain a model configuration and no fallback was supplied.")
        return fallback
    raw = dict(config["model"])
    experiment = config.get("experiment", {})
    experiment_id = experiment.get("experiment_id") if isinstance(experiment, dict) else None
    if "head_variant" not in raw and isinstance(experiment_id, str) and "probe42" in experiment_id:
        # v0.3.0 recorded the generic EfficientNet default even though the Probe42
        # experiment override built ConvNeXt. Normalize only this legacy shape.
        raw["backbone_name"] = "convnext_tiny"
        raw["head_variant"] = "canonical"
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{name: value for name, value in raw.items() if name in allowed})

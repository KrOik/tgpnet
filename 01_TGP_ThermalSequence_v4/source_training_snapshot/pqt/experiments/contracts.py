from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from torch import nn

from pqt.config.schema import ModelConfig, TrainConfig

if TYPE_CHECKING:
    from pqt.model.backbone import BackboneSpec
    from pqt.model.necks import NeckSpec


ALLOWED_EXPERIMENT_PARTS: dict[str, str] = {
    "backbone": "backbone.py",
    "neck": "neck.py",
    "head": "head.py",
    "calibrator": "calibrator.py",
    "loss": "losses.py",
    "config": "config.py",
}

PART_ALIASES: dict[str, str] = {
    "backbone": "backbone",
    "neck": "neck",
    "head": "head",
    "calibrator": "calibrator",
    "loss": "loss",
    "losses": "loss",
    "config": "config",
}

ALLOWED_STATUSES = {
    "draft",
    "running",
    "paused",
    "done",
    "abandoned",
    "merged",
}

BackboneBuilder = Callable[[ModelConfig], "BackboneSpec"]
NeckBuilder = Callable[[ModelConfig, int], "NeckSpec | nn.Module"]
HeadBuilder = Callable[[ModelConfig, int], nn.Module]
CalibratorBuilder = Callable[[ModelConfig], nn.Module]
TrainConfigApplier = Callable[[TrainConfig], TrainConfig | None]
LossBuilder = Callable[[TrainConfig], nn.Module]


@dataclass(slots=True)
class AssemblyOverrides:
    backbone_builder: BackboneBuilder | None = None
    neck_builder: NeckBuilder | None = None
    head_builder: HeadBuilder | None = None
    calibrator_builder: CalibratorBuilder | None = None


@dataclass(slots=True)
class ExperimentManifest:
    experiment_id: str
    version: str
    status: str
    branch: str
    baseline: str
    changed_parts: tuple[str, ...]
    config_path: Path | None
    override_paths: dict[str, Path]
    parent_experiment_id: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class ExperimentRuntime:
    manifest: ExperimentManifest
    experiment_dir: Path
    assembly_overrides: AssemblyOverrides = field(default_factory=AssemblyOverrides)
    config_applier: TrainConfigApplier | None = None
    loss_builder: LossBuilder | None = None


# The resolved object is the single experiment contract consumed by every workflow.
ResolvedExperiment = ExperimentRuntime

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from pqt.config.schema import ModelConfig
from pqt.experiments.contracts import AssemblyOverrides

from .backbone import BackboneSpec, MODEL_REGISTRY, adapt_first_conv
from .calibrators import build_calibrator
from .heads import build_head
from .necks import NeckSpec, build_neck


@dataclass(slots=True)
class ModelComponents:
    backbone: BackboneSpec
    neck: NeckSpec
    head: nn.Module
    calibrator: nn.Module

def assemble_model_components(
    model_config: ModelConfig,
    overrides: AssemblyOverrides | None = None,
) -> ModelComponents:
    backbone_builder = (
        overrides.backbone_builder
        if overrides and overrides.backbone_builder
        else MODEL_REGISTRY.get(model_config.backbone_name)
    )
    backbone_spec = backbone_builder(model_config)
    if overrides and overrides.neck_builder is not None:
        neck = overrides.neck_builder(model_config, backbone_spec.feature_dim)
        neck_spec = neck if isinstance(neck, NeckSpec) else NeckSpec(module=neck, output_dim=backbone_spec.feature_dim)
    else:
        neck_spec = build_neck(model_config, backbone_spec.feature_dim)
    head = (
        overrides.head_builder(model_config, neck_spec.output_dim)
        if overrides and overrides.head_builder
        else build_head(model_config, neck_spec.output_dim)
    )
    adapt_first_conv(backbone_spec.module, model_config.input_channels)
    calibrator = (
        overrides.calibrator_builder(model_config)
        if overrides and overrides.calibrator_builder
        else build_calibrator(model_config)
    )
    return ModelComponents(
        backbone=backbone_spec,
        neck=neck_spec,
        head=head,
        calibrator=calibrator,
    )

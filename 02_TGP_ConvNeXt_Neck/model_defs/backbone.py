from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from pqt.config.schema import ModelConfig
from pqt.model.backbone import (
    BackboneSpec,
    FeatureMapMetadata,
    build_efficientnet_v2_s_backbone,
    build_resnet50_backbone,
)

CAPTURE_INDICES = (1, 3, 5, 7)
STAGE_DIMS = (96, 192, 384, 768)
FEATURE_MAPS = (
    FeatureMapMetadata("C2", 1, 4, 96, (128, 128)),
    FeatureMapMetadata("P3", 3, 8, 192, (64, 64)),
    FeatureMapMetadata("P4", 5, 16, 384, (32, 32)),
    FeatureMapMetadata("P5", 7, 32, 768, (16, 16)),
)


def _set_depthwise_dilation(module: nn.Module, dilation: int) -> None:
    """Expand ConvNeXt's 7x7 depthwise receptive field without changing weights."""
    for layer in module.modules():
        if not isinstance(layer, nn.Conv2d):
            continue
        is_convnext_depthwise = (
            layer.kernel_size == (7, 7)
            and layer.stride == (1, 1)
            and layer.groups == layer.in_channels == layer.out_channels
        )
        if is_convnext_depthwise:
            layer.dilation = (dilation, dilation)
            layer.padding = (3 * dilation, 3 * dilation)


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
    return payload if isinstance(payload, dict) else {}


def _strip_prefixes(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "backbone.", "encoder.", "network."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    return normalized


def _load_local_checkpoint(module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    module.load_state_dict(_strip_prefixes(_extract_state_dict(payload)), strict=False)


def _extract_stage_features(backbone: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    features = x
    captured = []
    for index, block in enumerate(backbone.features):
        features = block(features)
        if index in backbone.capture_indices:
            captured.append(features)
    if len(captured) != len(STAGE_DIMS):
        raise RuntimeError(f"Expected {len(STAGE_DIMS)} ConvNeXt stages, got {len(captured)}.")
    return captured


def build_backbone(model_config: ModelConfig) -> BackboneSpec:
    if model_config.backbone_name == "efficientnet_v2_s":
        return build_efficientnet_v2_s_backbone(model_config)
    if model_config.backbone_name == "resnet50":
        return build_resnet50_backbone(model_config)
    if model_config.backbone_name != "convnext_tiny":
        raise ValueError(f"Unsupported Probe42 backbone: {model_config.backbone_name}")
    trunk = models.convnext_tiny(weights=None)
    dilation_schedule = model_config.convnext_stage_dilations or (model_config.convnext_dw_dilation,) * len(CAPTURE_INDICES)
    for feature_index, dilation in zip(CAPTURE_INDICES, dilation_schedule, strict=True):
        _set_depthwise_dilation(trunk.features[feature_index], dilation)
    trunk.capture_indices = CAPTURE_INDICES
    checkpoint_path = getattr(model_config, "pretrained_backbone_path", None)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        _load_local_checkpoint(trunk, Path(checkpoint_path))
    return BackboneSpec(
        module=trunk,
        feature_dim=sum(STAGE_DIMS),
        extract_features=_extract_stage_features,
        feature_maps=FEATURE_MAPS,
    )

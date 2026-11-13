from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision import models

from pqt.config.schema import ModelConfig
from pqt.registry.models import MODEL_REGISTRY

from .checkpoint import load_trusted_checkpoint_payload, unwrap_model_state_dict


@dataclass(slots=True)
class FeatureMapMetadata:
    name: str
    module_index: int
    stride: int
    channels: int
    spatial_size: tuple[int, int] | None = None


@dataclass(slots=True)
class BackboneSpec:
    module: nn.Module
    feature_dim: int
    extract_features: Callable[[nn.Module, torch.Tensor], torch.Tensor | list[torch.Tensor]]
    feature_maps: tuple[FeatureMapMetadata, ...] = ()


def adapt_first_conv(backbone: nn.Module, input_channels: int) -> None:
    direct_conv = getattr(backbone, "conv1", None)
    if isinstance(direct_conv, nn.Conv2d) and direct_conv.in_channels != input_channels:
        replacement = nn.Conv2d(
            input_channels,
            direct_conv.out_channels,
            kernel_size=direct_conv.kernel_size,
            stride=direct_conv.stride,
            padding=direct_conv.padding,
            bias=direct_conv.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.copy_(direct_conv.weight.mean(dim=1, keepdim=True))
            if direct_conv.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(direct_conv.bias)
        backbone.conv1 = replacement
        return
    features = getattr(backbone, "features", None)
    if features is None:
        return
    try:
        original_conv = features[0][0]
    except (IndexError, KeyError, TypeError):
        return
    if not isinstance(original_conv, nn.Conv2d):
        return
    if original_conv.in_channels == input_channels:
        return

    new_conv = nn.Conv2d(
        in_channels=input_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    with torch.no_grad():
        if input_channels == 1:
            new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        else:
            repeated = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
            new_conv.weight.copy_(repeated / float(input_channels))
        if original_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(original_conv.bias)

    features[0][0] = new_conv


def _load_local_pretrained(backbone: models.EfficientNet, checkpoint_path: Path) -> None:
    state_dict = unwrap_model_state_dict(load_trusted_checkpoint_payload(checkpoint_path, map_location="cpu"))
    if any(key.startswith("efficientnet.") for key in state_dict):
        state_dict = {
            key.replace("efficientnet.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("efficientnet.")
        }
    backbone.load_state_dict(state_dict, strict=False)


def _extract_efficientnet_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = backbone.features(x)
    features = backbone.avgpool(features)
    return torch.flatten(features, 1)


@MODEL_REGISTRY.register("efficientnet_v2_s")
def build_efficientnet_v2_s_backbone(model_config: ModelConfig) -> BackboneSpec:
    backbone = models.efficientnet_v2_s(weights=None)
    if model_config.pretrained and model_config.pretrained_backbone_path is not None:
        pretrained_path = Path(model_config.pretrained_backbone_path)
        if pretrained_path.exists():
            _load_local_pretrained(backbone, pretrained_path)
    feature_dim = backbone.classifier[1].in_features
    return BackboneSpec(
        module=backbone,
        feature_dim=feature_dim,
        extract_features=_extract_efficientnet_features,
    )


def _extract_resnet_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return torch.flatten(backbone.avgpool(x), 1)


@MODEL_REGISTRY.register("resnet50")
def build_resnet50_backbone(model_config: ModelConfig) -> BackboneSpec:
    backbone = models.resnet50(weights=None)
    return BackboneSpec(module=backbone, feature_dim=backbone.fc.in_features, extract_features=_extract_resnet_features)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from pqt.config.schema import ModelConfig
from pqt.registry.models import MODEL_REGISTRY

from .checkpoint import load_trusted_checkpoint_payload, unwrap_model_state_dict
from .blocks import SEBlock


@dataclass(slots=True)
class FeatureMapMetadata:
    name: str
    module_index: int
    stride: int
    channels: int
    spatial_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class FeaturePyramid:
    """Named four-scale output produced by a TGPNet encoder.

    The named fields make the encoder contract explicit while ``as_tuple``
    keeps the existing necks, which consume ordered feature lists, compatible.
    """

    c2: torch.Tensor
    p3: torch.Tensor
    p4: torch.Tensor
    p5: torch.Tensor

    @property
    def names(self) -> tuple[str, str, str, str]:
        return ("C2", "P3", "P4", "P5")

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.c2, self.p3, self.p4, self.p5)

    def __iter__(self):
        return iter(self.as_tuple())

    @classmethod
    def from_sequence(cls, features: Iterable[torch.Tensor]) -> "FeaturePyramid":
        values = tuple(features)
        if len(values) != 4:
            raise ValueError(f"A TGPNet feature pyramid requires four maps, got {len(values)}.")
        return cls(*values)


@dataclass(slots=True)
class BackboneSpec:
    module: nn.Module
    feature_dim: int
    extract_features: Callable[[nn.Module, torch.Tensor], torch.Tensor | list[torch.Tensor] | FeaturePyramid]
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


def _load_local_pretrained(backbone: nn.Module, checkpoint_path: Path) -> None:
    state_dict = unwrap_model_state_dict(load_trusted_checkpoint_payload(checkpoint_path, map_location="cpu"))
    if any(key.startswith("efficientnet.") for key in state_dict):
        state_dict = {
            key.replace("efficientnet.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("efficientnet.")
        }
    backbone.load_state_dict(state_dict, strict=True)


class SRAInputEncoder(nn.Module):
    """Encode a normalized thermal image as intensity/log/intensity channels.

    The dataset supplies a single channel normalized to ``[-1, 1]``.  The
    encoder restores ``[0, 1]`` intensity, adds a log-enhanced channel, and
    applies ImageNet normalization so the original ConvNeXt stem remains
    compatible with its pretrained weights.
    """

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.tensor(self.imagenet_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.imagenet_std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SRAInputEncoder expects NCHW input, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            return x
        if x.shape[1] != 1:
            raise ValueError(f"SRAInputEncoder expects one or three channels, got {x.shape[1]}")
        intensity = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        if self.enabled:
            log_enhanced = torch.log1p(255.0 * intensity) / torch.log(x.new_tensor(256.0))
            encoded = torch.cat((intensity, log_enhanced, intensity), dim=1)
        else:
            encoded = intensity.expand(-1, 3, -1, -1)
        return (encoded - self.mean.to(dtype=encoded.dtype)) / self.std.to(dtype=encoded.dtype)


class TopDownSERefinement(nn.Module):
    """Additive top-down interaction with per-scale channel recalibration.

    This is deliberately deterministic additive fusion: there is no scale
    softmax, routing probability, expert selection, or learned scale gate.
    """

    channels = (96, 192, 384, 768)

    def __init__(self, *, cross_scale: bool = True, se: bool = True) -> None:
        super().__init__()
        self.cross_scale = cross_scale
        self.se = se
        self.top_down = nn.ModuleList([
            nn.Conv2d(source, target, kernel_size=1, bias=False)
            for source, target in zip(self.channels[:0:-1], self.channels[-2::-1], strict=True)
        ])
        self.recalibration = nn.ModuleList([
            SEBlock(width) if se else nn.Identity() for width in self.channels
        ])

    def forward(self, pyramid: FeaturePyramid) -> FeaturePyramid:
        n2, n3, n4, n5 = pyramid.as_tuple()
        p5 = self.recalibration[3](n5)
        if not self.cross_scale:
            return FeaturePyramid(
                self.recalibration[0](n2), self.recalibration[1](n3),
                self.recalibration[2](n4), p5,
            )
        p4 = self.recalibration[2](n4 + F.interpolate(
            self.top_down[0](p5), size=n4.shape[-2:], mode="bilinear", align_corners=False,
        ))
        p3 = self.recalibration[1](n3 + F.interpolate(
            self.top_down[1](p4), size=n3.shape[-2:], mode="bilinear", align_corners=False,
        ))
        p2 = self.recalibration[0](n2 + F.interpolate(
            self.top_down[2](p3), size=n2.shape[-2:], mode="bilinear", align_corners=False,
        ))
        return FeaturePyramid(p2, p3, p4, p5)


class TGPNetConvNeXtBackbone(nn.Module):
    """Pretrained ConvNeXt-Tiny with SRA encoding and a four-level contract."""

    stage_indices = (1, 3, 5, 7)

    def __init__(
        self,
        *,
        pretrained: bool,
        pretrained_path: Path | None,
        sra_encoding: bool = True,
        cross_scale_refinement: bool = True,
        se_refinement: bool = True,
    ) -> None:
        super().__init__()
        self.input_encoder = SRAInputEncoder(enabled=sra_encoding)
        self.encoder = models.convnext_tiny(weights=None)
        if pretrained:
            if pretrained_path is None:
                raise FileNotFoundError("A local pretrained ConvNeXt-Tiny checkpoint is required.")
            checkpoint = Path(pretrained_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Pretrained ConvNeXt-Tiny checkpoint not found: {checkpoint}")
            _load_local_pretrained(self.encoder, checkpoint)
        self.refinement = TopDownSERefinement(
            cross_scale=cross_scale_refinement,
            se=se_refinement,
        )
        self.capture_indices = self.stage_indices
        self.expects_dual_resolution = False

    def forward_features(self, x: torch.Tensor) -> FeaturePyramid:
        x = self.input_encoder(x)
        features = self.encoder.features
        if len(features) != 8:
            raise RuntimeError(f"Unexpected torchvision ConvNeXt feature layout: {len(features)} modules")
        x = features[0](x)
        n2 = features[1](x)
        x = features[2](n2)
        n3 = features[3](x)
        x = features[4](n3)
        n4 = features[5](x)
        x = features[6](n4)
        n5 = features[7](x)
        return self.refinement(FeaturePyramid(n2, n3, n4, n5))

    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        return self.forward_features(x)

    def diagnostic_component_paths(self) -> dict[str, str]:
        return {
            "backbone_sra_encoder": "model.backbone.input_encoder",
            "backbone_cross_scale_refinement": "model.backbone.refinement",
            "backbone_top_down_projection_0": "model.backbone.refinement.top_down.0",
            "backbone_top_down_projection_1": "model.backbone.refinement.top_down.1",
            "backbone_top_down_projection_2": "model.backbone.refinement.top_down.2",
        }


def _extract_efficientnet_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = backbone.features(x)
    features = backbone.avgpool(features)
    return torch.flatten(features, 1)


def _extract_tgp_convnext_features(backbone: nn.Module, x: torch.Tensor) -> FeaturePyramid:
    return backbone.forward_features(x)


@MODEL_REGISTRY.register("tgp_pqt_convnext_v1")
def build_tgp_pqt_convnext_backbone(model_config: ModelConfig) -> BackboneSpec:
    checkpoint_path = model_config.pretrained_backbone_path
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parents[2] / "artifacts" / "pretrained" / "convnext_tiny_imagenet1k_v1.pth"
    backbone = TGPNetConvNeXtBackbone(
        pretrained=model_config.pretrained,
        pretrained_path=checkpoint_path,
        sra_encoding=model_config.sra_encoding,
        cross_scale_refinement=model_config.cross_scale_refinement,
        se_refinement=model_config.se_refinement,
    )
    feature_maps = tuple(
        FeatureMapMetadata(name=name, module_index=index, stride=stride, channels=channels)
        for name, index, stride, channels in zip(
            ("C2", "P3", "P4", "P5"), (1, 3, 5, 7), (4, 8, 16, 32), (96, 192, 384, 768), strict=True
        )
    )
    return BackboneSpec(
        module=backbone,
        feature_dim=768,
        extract_features=_extract_tgp_convnext_features,
        feature_maps=feature_maps,
    )


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


def _extract_thermo_seq_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return backbone(x)


@MODEL_REGISTRY.register("tgp_thermo_seq_v1")
def build_tgp_thermo_seq_backbone(model_config: ModelConfig) -> BackboneSpec:
    from .thermo import TGPThermoSeqBackbone

    backbone = TGPThermoSeqBackbone(
        scales=model_config.thermal_scales,
        channels=model_config.thermal_stage_channels,
        segments=model_config.thermal_pool_segments,
        embed_dim=model_config.thermal_embed_dim,
        multiscale_encoding=model_config.thermal_multiscale_encoding,
        include_raw=model_config.thermal_include_raw,
        position_features=model_config.thermal_position_features,
    )
    return BackboneSpec(
        module=backbone,
        feature_dim=backbone.embed_dim,
        extract_features=_extract_thermo_seq_features,
    )


@MODEL_REGISTRY.register("tgp_thermo_seq_v2")
def build_tgp_thermo_seq_v2_backbone(model_config: ModelConfig) -> BackboneSpec:
    from .thermo import TGPThermoSeqBackbone

    backbone = TGPThermoSeqBackbone(
        scales=model_config.thermal_scales,
        channels=model_config.thermal_stage_channels,
        segments=model_config.thermal_pool_segments,
        embed_dim=model_config.thermal_embed_dim,
        multiscale_encoding=model_config.thermal_multiscale_encoding,
        include_raw=model_config.thermal_include_raw,
        position_features=model_config.thermal_position_features,
        soft_position=model_config.thermal_soft_position,
        global_stats=model_config.thermal_global_stats,
        spectral_bins=model_config.thermal_spectral_bins,
        dropout=model_config.thermal_dropout,
    )
    if model_config.pretrained and model_config.pretrained_backbone_path is not None:
        _load_local_pretrained(backbone, Path(model_config.pretrained_backbone_path))
    return BackboneSpec(
        module=backbone,
        feature_dim=backbone.embed_dim,
        extract_features=_extract_thermo_seq_features,
    )


@MODEL_REGISTRY.register("tgp_thermo_geo_v1")
def build_tgp_thermo_geo_backbone(model_config: ModelConfig) -> BackboneSpec:
    from .thermo import TGPThermoGeoBackbone

    backbone = TGPThermoGeoBackbone(
        context_enabled=model_config.thermal_context_enabled,
        context_dim=model_config.thermal_context_dim,
        ring_edges=model_config.thermal_context_ring_edges,
        scales=model_config.thermal_scales,
        channels=model_config.thermal_stage_channels,
        segments=model_config.thermal_pool_segments,
        embed_dim=model_config.thermal_embed_dim,
        multiscale_encoding=model_config.thermal_multiscale_encoding,
        include_raw=model_config.thermal_include_raw,
        position_features=model_config.thermal_position_features,
    )
    return BackboneSpec(
        module=backbone,
        feature_dim=backbone.embed_dim,
        extract_features=_extract_thermo_seq_features,
    )

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pqt.config.schema import ModelConfig
from pqt.registry.necks import NECK_REGISTRY

from .blocks import SEBlock


@dataclass(slots=True)
class NeckSpec:
    module: nn.Module
    output_dim: int


class IdentityNeck(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SqueezeExcitationNeck(nn.Module):
    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super().__init__()
        self.se_block = SEBlock(in_features, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se_block(x.unsqueeze(-1).unsqueeze(-1)).flatten(1)


class PyramidDescriptor(nn.Module):
    """Convert C2/P3/P4/P5 into an equal-width, order-preserving descriptor."""

    input_channels = (96, 192, 384, 768)

    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, projection_dim, kernel_size=1, bias=False)
            for channels in self.input_channels
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norms = nn.ModuleList(nn.LayerNorm(projection_dim) for _ in self.input_channels)
        self.output_dim = len(self.input_channels) * projection_dim

    def forward(self, features) -> torch.Tensor:
        if hasattr(features, "as_tuple"):
            features = features.as_tuple()
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("PyramidDescriptor requires C2/P3/P4/P5 in that order.")
        descriptors = [
            norm(self.pool(projection(feature)).flatten(1))
            for projection, norm, feature in zip(self.projections, self.norms, features, strict=True)
        ]
        return torch.cat(descriptors, dim=1)


@NECK_REGISTRY.register("identity")
def build_identity_neck(_: ModelConfig, input_dim: int) -> NeckSpec:
    return NeckSpec(module=IdentityNeck(), output_dim=input_dim)


@NECK_REGISTRY.register("se_attention")
def build_se_attention_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    return NeckSpec(
        module=SqueezeExcitationNeck(input_dim, reduction=model_config.se_reduction),
        output_dim=input_dim,
    )


@NECK_REGISTRY.register("pyramid_descriptor")
def build_pyramid_descriptor_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    del input_dim
    descriptor = PyramidDescriptor(projection_dim=model_config.multiscale_projection_dim)
    return NeckSpec(module=descriptor, output_dim=descriptor.output_dim)


def build_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    neck_builder = NECK_REGISTRY.get(model_config.neck_name)
    neck = neck_builder(model_config, input_dim)
    if isinstance(neck, NeckSpec):
        return neck
    return NeckSpec(module=neck, output_dim=input_dim)

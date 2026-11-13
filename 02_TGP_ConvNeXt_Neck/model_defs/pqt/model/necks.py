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


@NECK_REGISTRY.register("identity")
def build_identity_neck(_: ModelConfig, input_dim: int) -> NeckSpec:
    return NeckSpec(module=IdentityNeck(), output_dim=input_dim)


@NECK_REGISTRY.register("se_attention")
def build_se_attention_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    return NeckSpec(
        module=SqueezeExcitationNeck(input_dim, reduction=model_config.se_reduction),
        output_dim=input_dim,
    )


def build_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    neck_builder = NECK_REGISTRY.get(model_config.neck_name)
    neck = neck_builder(model_config, input_dim)
    if isinstance(neck, NeckSpec):
        return neck
    return NeckSpec(module=neck, output_dim=input_dim)

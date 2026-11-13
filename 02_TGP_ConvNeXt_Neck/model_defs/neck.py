from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pqt.config.schema import ModelConfig
from pqt.model.necks import NeckSpec
from pqt.observation import publish


INPUT_CHANNELS = (96, 192, 384, 768)
FUSION_CHANNELS = 160
OUTPUT_DIM = len(INPUT_CHANNELS) * FUSION_CHANNELS
NORMALIZED_SCALE_DIM = 128
NORMALIZED_OUTPUT_DIM = len(INPUT_CHANNELS) * NORMALIZED_SCALE_DIM


class GeMPool2d(nn.Module):
    def __init__(self, initial_p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([initial_p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        pooled = F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), output_size=1)
        return pooled.pow(p.reciprocal())


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ScaleGatedPooling(nn.Module):
    """Pool and combine the four canonical ConvNeXt feature scales."""

    def __init__(self, channels: int, num_scales: int, policy: str = "learned_softmax") -> None:
        super().__init__()
        if policy not in {"learned_softmax", "fixed_uniform"}:
            raise ValueError(f"Unsupported scale gating policy: {policy}")
        self.policy = policy
        self.num_scales = num_scales
        self.pool = GeMPool2d()
        self.refine = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU())
            for _ in range(num_scales)
        ])
        self.gates = (
            nn.ModuleList([
                nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 1))
                for _ in range(num_scales)
            ])
            if policy == "learned_softmax"
            else None
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pooled = [self.pool(feature).flatten(1) for feature in features]
        pooled = [refine(vector) + vector for refine, vector in zip(self.refine, pooled, strict=True)]
        if self.gates is None:
            gate_values = pooled[0].new_full((pooled[0].shape[0], self.num_scales), 1.0 / self.num_scales)
        else:
            gate_logits = [gate(vector) for gate, vector in zip(self.gates, pooled, strict=True)]
            gate_values = torch.softmax(torch.cat(gate_logits, dim=1), dim=1)
        publish("scale_gate_probs", gate_values)
        publish("scale_feature_norms", torch.stack([vector.norm(dim=1) for vector in pooled], dim=1))
        gated = [vector * gate_values[:, index : index + 1] for index, vector in enumerate(pooled)]
        return torch.cat(gated, dim=1)


def _make_projection(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class PhysicsBandBlock(nn.Module):
    """Separate smooth diffusion and edge bands, then learn their recomposition."""

    def __init__(self, channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.diffusion = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            DepthwiseResidualBlock(channels, dilation=dilation),
        )
        self.edge = DepthwiseResidualBlock(channels)
        self.summary_pool = GeMPool2d()
        self.recompose = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffusion = self.diffusion(x)
        edge = self.edge(x - diffusion)
        diff_desc = F.normalize(self.summary_pool(diffusion).flatten(1), dim=1)
        edge_desc = F.normalize(self.summary_pool(edge).flatten(1), dim=1)
        edge_desc = edge_desc - (edge_desc * diff_desc).sum(dim=1, keepdim=True) * diff_desc
        weights = self.recompose(torch.cat([diff_desc, edge_desc], dim=1)).unsqueeze(-1).unsqueeze(-1)
        publish("physics_diffusion_energy", diffusion.square().mean(dim=(1, 2, 3)))
        publish("physics_edge_energy", edge.square().mean(dim=(1, 2, 3)))
        publish("physics_band_correlation", (diff_desc * edge_desc).sum(dim=1))
        publish("physics_recompose_weight", weights.flatten(1))
        return diffusion * (1.0 - weights) + edge * weights


class PhysicsBandDecoupleNeck(nn.Module):
    def __init__(
        self,
        scale_gating_policy: str = "learned_softmax",
        physics_band_dilation: int = 2,
        physics_band_dilations: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        dilation_schedule = physics_band_dilations or (physics_band_dilation,) * len(INPUT_CHANNELS)
        self.projections = nn.ModuleList([_make_projection(channels, FUSION_CHANNELS) for channels in INPUT_CHANNELS])
        self.blocks = nn.ModuleList([
            PhysicsBandBlock(FUSION_CHANNELS, dilation=dilation)
            for dilation in dilation_schedule
        ])
        self.pooling = ScaleGatedPooling(
            FUSION_CHANNELS, len(INPUT_CHANNELS), policy=scale_gating_policy
        )

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} feature maps, got {type(features)!r}.")
        refined = [
            block(projection(feature))
            for projection, block, feature in zip(self.projections, self.blocks, features, strict=True)
        ]
        return self.pooling(refined)


class MultiScaleFeatureSelect(nn.Module):
    """Pass only P3/P4/P5 to the independent scalar branches; C2 remains audit-only."""

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("multi_scale requires the four audited ConvNeXt feature maps.")
        return tuple(features[1:])


class NormalizedFourScaleDescriptor(nn.Module):
    """Project, pool, and independently normalize all four ConvNeXt stages."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, NORMALIZED_SCALE_DIM, kernel_size=1, bias=False)
            for channels in INPUT_CHANNELS
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norms = nn.ModuleList(nn.LayerNorm(NORMALIZED_SCALE_DIM) for _ in INPUT_CHANNELS)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.projections):
            raise ValueError("normalized_four_scale requires all four ConvNeXt feature maps.")
        pooled = [
            self.pool(projection(feature)).flatten(1)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        normalized = [norm(vector) for norm, vector in zip(self.norms, pooled, strict=True)]
        publish("normalized_scale_descriptor_norms_raw", torch.stack([vector.norm(dim=1) for vector in pooled], dim=1))
        publish(
            "normalized_scale_descriptor_norms_post",
            torch.stack([vector.norm(dim=1) for vector in normalized], dim=1),
        )
        descriptor = torch.cat(normalized, dim=1)
        if descriptor.shape[1] != NORMALIZED_OUTPUT_DIM:
            raise RuntimeError(f"normalized_four_scale descriptor must be {NORMALIZED_OUTPUT_DIM}-D.")
        return descriptor


class DirectFourScalePerScaleNormDescriptor(nn.Module):
    """B1 descriptor with the sole intervention of one LayerNorm per scale."""

    descriptor_dim = NORMALIZED_SCALE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, self.descriptor_dim, kernel_size=1, bias=False)
            for channels in INPUT_CHANNELS
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norms = nn.ModuleList(nn.LayerNorm(self.descriptor_dim) for _ in INPUT_CHANNELS)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.projections):
            raise ValueError("direct_four_scale_per_scale_norm requires all four ConvNeXt feature maps.")
        pooled = [
            self.pool(projection(feature)).flatten(1)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        normalized = [norm(vector) for norm, vector in zip(self.norms, pooled, strict=True)]
        publish("per_scale_norm_descriptor_norms_raw", torch.stack([vector.norm(dim=1) for vector in pooled], dim=1))
        publish("per_scale_norm_descriptor_norms_post", torch.stack([vector.norm(dim=1) for vector in normalized], dim=1))
        return torch.cat(normalized, dim=1)


class DirectFourScaleDescriptor(nn.Module):
    """Locked current-code reproduction of the historical B1 descriptor."""

    descriptor_dim = NORMALIZED_SCALE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, self.descriptor_dim, kernel_size=1, bias=False)
            for channels in INPUT_CHANNELS
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.projections):
            raise ValueError("direct_four_scale requires all four ConvNeXt feature maps.")
        descriptor = torch.cat(
            [
                self.pool(projection(feature)).flatten(1)
                for projection, feature in zip(self.projections, features, strict=True)
            ],
            dim=1,
        )
        if descriptor.shape[1] != NORMALIZED_OUTPUT_DIM:
            raise RuntimeError("direct_four_scale descriptor must be 512-D.")
        publish("b1_descriptor_shape", torch.tensor(descriptor.shape, device=descriptor.device))
        return descriptor


def build_neck(model_config: ModelConfig, input_dim: int) -> NeckSpec:
    if model_config.backbone_name in {"efficientnet_v2_s", "resnet50"}:
        return NeckSpec(module=nn.Identity(), output_dim=input_dim)
    if model_config.head_variant == "multi_scale":
        return NeckSpec(module=MultiScaleFeatureSelect(), output_dim=sum(INPUT_CHANNELS[1:]))
    if model_config.head_variant == "normalized_four_scale":
        return NeckSpec(module=NormalizedFourScaleDescriptor(), output_dim=NORMALIZED_OUTPUT_DIM)
    if model_config.head_variant == "direct_four_scale":
        return NeckSpec(module=DirectFourScaleDescriptor(), output_dim=NORMALIZED_OUTPUT_DIM)
    if model_config.head_variant == "direct_four_scale_per_scale_norm":
        return NeckSpec(module=DirectFourScalePerScaleNormDescriptor(), output_dim=NORMALIZED_OUTPUT_DIM)
    del input_dim
    return NeckSpec(
        module=PhysicsBandDecoupleNeck(
            model_config.scale_gating_policy,
            physics_band_dilation=model_config.physics_band_dilation,
            physics_band_dilations=model_config.physics_band_dilations,
        ),
        output_dim=OUTPUT_DIM,
    )

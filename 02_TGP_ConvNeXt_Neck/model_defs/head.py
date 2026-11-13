from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from pqt.config.schema import ModelConfig
from pqt.depth_ordinal import DEPTH_LEVELS, MAX_RESIDUAL, decode_ordinal_logits
from pqt.model.outputs import RegressionOutput

ROUTING_TEMPERATURE = 0.85
NUM_EXPERTS = 4
MULTI_SCALE_CHANNELS = (192, 384, 768)
NORMALIZED_FOUR_SCALE_DIM = 512
NORMALIZED_BOTTLENECK_DIM = 256


class TopologyRoutingRegressionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        bottleneck = max(192, hidden_dim // 2)
        self.input_norm = nn.LayerNorm(in_features)
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
        )
        self.skip = nn.Linear(in_features, bottleneck)
        self.router = nn.Linear(bottleneck, NUM_EXPERTS)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(bottleneck),
                    nn.Linear(bottleneck, 96),
                    nn.GELU(),
                    nn.Linear(96, 1),
                )
                for _ in range(NUM_EXPERTS)
            ]
        )
        self.confidence = nn.Sequential(
            nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, 1),
        )
    def forward(self, x: torch.Tensor) -> RegressionOutput:
        normalized = self.input_norm(x)
        shared = self.shared(normalized) + self.skip(normalized)
        routing_logits = self.router(shared)
        routing_probs = torch.softmax(routing_logits / ROUTING_TEMPERATURE, dim=1)
        expert_outputs = torch.cat([expert(shared) for expert in self.experts], dim=1)
        prediction = (routing_probs * expert_outputs).sum(dim=1, keepdim=True)

        auxiliary = {
            "mode": "topology_routing",
            "routing_logits": routing_logits,
            "routing_probs": routing_probs,
            "expert_outputs": expert_outputs,
            "confidence": self.confidence(shared),
        }
        return RegressionOutput(prediction=prediction, auxiliary=auxiliary)


class GeMPool2d(nn.Module):
    def __init__(self, initial_p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([initial_p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(1.0, 6.0)
        pooled = F.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(p), 1)
        return pooled.pow(p.reciprocal()).flatten(1)


class ScalarMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = max(64, min(hidden_dim, 512))
        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OrdinalResidualHead(nn.Module):
    """Single-branch ordinal locator plus bounded continuous residual.

    This candidate intentionally has no router or experts.  It isolates the
    contribution of the nine-level target structure on the same canonical
    multiscale physics-band representation used by the clean scalar baseline.
    """

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = max(64, min(hidden_dim, 512))
        self.shared = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
        )
        self.ordinal_score = nn.Linear(hidden, 1)
        initial_gap = 4.0 / 7.0
        inverse_softplus_gap = torch.log(torch.expm1(torch.tensor(initial_gap)))
        self.cutpoint_parameters = nn.Parameter(
            torch.cat((torch.tensor([-2.0]), inverse_softplus_gap.repeat(7)))
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _cutpoints(self) -> torch.Tensor:
        first = self.cutpoint_parameters[:1]
        gaps = F.softplus(self.cutpoint_parameters[1:])
        return torch.cat((first, first + torch.cumsum(gaps, dim=0)))

    def forward(self, x: torch.Tensor) -> RegressionOutput:
        shared = self.shared(x)
        score = self.ordinal_score(shared)
        cutpoints = self._cutpoints()
        ordinal_logits = score - cutpoints.view(1, -1)
        cumulative, probabilities, coarse = decode_ordinal_logits(ordinal_logits, DEPTH_LEVELS)
        residual = MAX_RESIDUAL * torch.tanh(self.residual(shared))
        return RegressionOutput(
            prediction=coarse + residual,
            auxiliary={
                "mode": "ordinal_residual_single_head",
                "ordinal_score": score,
                "cutpoints": cutpoints,
                "ordinal_logits": ordinal_logits,
                "cumulative_probabilities": cumulative,
                "level_probabilities": probabilities,
                "coarse_depth": coarse,
                "residual": residual,
            },
        )


class NormalizedFourScaleHead(nn.Module):
    """Unbounded scalar regressor with a training-only ordinal auxiliary."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        if in_features != NORMALIZED_FOUR_SCALE_DIM:
            raise ValueError(
                f"normalized_four_scale requires a {NORMALIZED_FOUR_SCALE_DIM}-D descriptor, got {in_features}."
            )
        self.shared = nn.Sequential(
            nn.Linear(in_features, NORMALIZED_BOTTLENECK_DIM),
            nn.GELU(),
        )
        self.continuous = nn.Linear(NORMALIZED_BOTTLENECK_DIM, 1)
        self.ordinal_score = nn.Linear(NORMALIZED_BOTTLENECK_DIM, 1)
        initial_gap = 4.0 / 7.0
        inverse_softplus_gap = torch.log(torch.expm1(torch.tensor(initial_gap)))
        self.cutpoint_parameters = nn.Parameter(
            torch.cat((torch.tensor([-2.0]), inverse_softplus_gap.repeat(7)))
        )

    def _cutpoints(self) -> torch.Tensor:
        first = self.cutpoint_parameters[:1]
        gaps = F.softplus(self.cutpoint_parameters[1:])
        return torch.cat((first, first + torch.cumsum(gaps, dim=0)))

    def forward(self, x: torch.Tensor) -> RegressionOutput:
        shared = self.shared(x)
        prediction = self.continuous(shared)
        if not self.training:
            return RegressionOutput(prediction=prediction)
        score = self.ordinal_score(shared)
        cutpoints = self._cutpoints()
        return RegressionOutput(
            prediction=prediction,
            auxiliary={
                "mode": "normalized_four_scale_auxiliary",
                "ordinal_score": score,
                "cutpoints": cutpoints,
                "ordinal_logits": score - cutpoints.view(1, -1),
            },
        )


class MultiScaleDepthHead(nn.Module):
    def __init__(self, projection_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, projection_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(projection_dim),
                nn.GELU(),
            )
            for channels in MULTI_SCALE_CHANNELS
        )
        self.pools = nn.ModuleList(GeMPool2d() for _ in MULTI_SCALE_CHANNELS)
        self.branches = nn.ModuleList(ScalarMLP(projection_dim, hidden_dim) for _ in MULTI_SCALE_CHANNELS)
        self.fusion_logits = nn.Parameter(torch.zeros(len(MULTI_SCALE_CHANNELS)))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> RegressionOutput:
        if not isinstance(features, (list, tuple)) or len(features) != len(MULTI_SCALE_CHANNELS):
            raise ValueError("multi_scale head expects P3, P4 and P5 feature maps.")
        predictions = torch.cat(
            [
                branch(pool(projection(feature)))
                for projection, pool, branch, feature in zip(
                    self.projections, self.pools, self.branches, features, strict=True
                )
            ],
            dim=1,
        )
        weights = torch.softmax(self.fusion_logits, dim=0)
        prediction = (predictions * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        return RegressionOutput(
            prediction=prediction,
            auxiliary={
                "mode": "multi_scale",
                "branch_predictions": predictions,
                "fusion_logits": self.fusion_logits,
                "fusion_weights": weights,
            },
        )


def build_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    if model_config.backbone_name in {"efficientnet_v2_s", "resnet50"}:
        return ScalarMLP(input_dim, model_config.hidden_dim)
    if model_config.head_variant == "scalar_mlp":
        return ScalarMLP(input_dim, model_config.hidden_dim)
    if model_config.head_variant == "ordinal_residual":
        return OrdinalResidualHead(input_dim, model_config.hidden_dim)
    if model_config.head_variant == "normalized_four_scale":
        return NormalizedFourScaleHead(input_dim)
    if model_config.head_variant in {"direct_four_scale", "direct_four_scale_per_scale_norm"}:
        if input_dim != NORMALIZED_FOUR_SCALE_DIM:
            raise ValueError(f"{model_config.head_variant} requires a 512-D descriptor.")
        return ScalarMLP(input_dim, model_config.hidden_dim)
    if model_config.head_variant == "multi_scale":
        return MultiScaleDepthHead(model_config.multiscale_projection_dim, model_config.hidden_dim)
    if model_config.head_variant != "canonical":
        raise ValueError(f"Unsupported Probe42 head variant: {model_config.head_variant}")
    hidden_dim = max(model_config.hidden_dim, 896)
    return TopologyRoutingRegressionHead(in_features=input_dim, hidden_dim=hidden_dim)

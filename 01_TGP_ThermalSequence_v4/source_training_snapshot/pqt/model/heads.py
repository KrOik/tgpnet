from __future__ import annotations

import torch
from torch import nn

from pqt.config.schema import ModelConfig
from pqt.registry.heads import HEAD_REGISTRY


class ResidualRegressionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 512, num_outputs: int = 1) -> None:
        super().__init__()
        bottleneck = max(16, hidden_dim // 4)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(hidden_dim, bottleneck)
        self.norm2 = nn.LayerNorm(bottleneck)
        self.relu2 = nn.ReLU(inplace=True)
        self.skip_connection = nn.Linear(in_features, bottleneck)
        self.regression_layer = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_outputs),
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.dropout1(x)
        main = self.linear1(main)
        main = self.norm1(main)
        main = self.relu1(main)
        main = self.dropout2(main)
        main = self.linear2(main)
        main = self.norm2(main)
        main = self.relu2(main)
        skip = self.skip_connection(x)
        output = self.regression_layer(main + skip)
        return output * self.scale + self.shift


class PQTLinearRegressionHead(nn.Module):
    """Parameter-matched linear readout used as the RRH ablation."""

    def __init__(self, in_features: int, num_outputs: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@HEAD_REGISTRY.register("residual_regression")
def build_residual_regression_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    return ResidualRegressionHead(
        in_features=input_dim,
        hidden_dim=model_config.hidden_dim,
        num_outputs=model_config.num_outputs,
    )


@HEAD_REGISTRY.register("pqt_rrh")
def build_pqt_rrh_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    return ResidualRegressionHead(
        in_features=input_dim,
        hidden_dim=model_config.hidden_dim,
        num_outputs=model_config.num_outputs,
    )


@HEAD_REGISTRY.register("pqt_linear")
def build_pqt_linear_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    return PQTLinearRegressionHead(input_dim, num_outputs=model_config.num_outputs)


def build_head(model_config: ModelConfig, input_dim: int) -> nn.Module:
    head_builder = HEAD_REGISTRY.get(model_config.head_name)
    return head_builder(model_config, input_dim)

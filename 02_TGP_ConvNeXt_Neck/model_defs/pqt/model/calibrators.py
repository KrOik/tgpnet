from __future__ import annotations

import torch
from torch import nn

from pqt.config.schema import ModelConfig
from pqt.registry.calibrators import CALIBRATOR_REGISTRY


class IdentityCalibrator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TemperatureCalibrator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.temperature.clamp_min(1e-6)


@CALIBRATOR_REGISTRY.register("identity")
def build_identity_calibrator(_: ModelConfig) -> nn.Module:
    return IdentityCalibrator()


@CALIBRATOR_REGISTRY.register("temperature")
def build_temperature_calibrator(_: ModelConfig) -> nn.Module:
    return TemperatureCalibrator()


def build_calibrator(model_config: ModelConfig) -> nn.Module:
    calibrator_builder = CALIBRATOR_REGISTRY.get(model_config.calibration_name)
    return calibrator_builder(model_config)

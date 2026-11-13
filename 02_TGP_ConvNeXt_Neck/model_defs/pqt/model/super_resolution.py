from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pqt.assets import file_sha256
from pqt.data.sr_cache import SR_WEIGHT_SHA256, SR_WEIGHT_SIZE
from pqt.model.inputs import DualResolutionInput


class FrozenEDSRx2(nn.Module):
    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        from torchsr.models import edsr_baseline

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing EDSR x2 weights: {weights_path}")
        if weights_path.stat().st_size != SR_WEIGHT_SIZE or file_sha256(weights_path) != SR_WEIGHT_SHA256:
            raise ValueError("EDSR x2 weight identity mismatch.")
        self.model = edsr_baseline(scale=2, pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.model.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, original: torch.Tensor) -> DualResolutionInput:
        rgb = original.mul(0.5).add(0.5).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        with torch.no_grad():
            sr_rgb = self.model(rgb).float().clamp_(0.0, 1.0)
            luma = 0.299 * sr_rgb[:, 0:1] + 0.587 * sr_rgb[:, 1:2] + 0.114 * sr_rgb[:, 2:3]
        return DualResolutionInput(original, luma.sub(0.5).div(0.5))

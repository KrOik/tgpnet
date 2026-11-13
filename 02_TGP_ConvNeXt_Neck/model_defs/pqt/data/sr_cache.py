from __future__ import annotations

from pathlib import Path

SR_MODEL_ID = "torchsr-edsr-baseline-x2"
SR_MODEL_VERSION = "1.0.3"
SR_SCALE = 2
SR_INPUT_SIZE = (1024, 1024)
SR_OUTPUT_SIZE = (2048, 2048)
SR_WEIGHT_SIZE = 5_491_291
SR_WEIGHT_SHA256 = "2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1"
SR_WEIGHT_RELATIVE_PATH = Path("artifacts/pretrained/edsr64_x2.pt")

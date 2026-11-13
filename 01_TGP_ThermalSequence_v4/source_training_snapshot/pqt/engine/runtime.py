from __future__ import annotations

import os
import random

import numpy as np
import torch

from pqt.config.schema import RuntimeConfig


def seed_runtime(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # PyTorch 2.6 has no strict deterministic CUDA backward for
        # adaptive_avg_pool2d. Keep every available deterministic control while
        # allowing that documented fallback instead of aborting training.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(runtime_config: RuntimeConfig) -> torch.device:
    if runtime_config.device == "cpu":
        return torch.device("cpu")
    if runtime_config.device == "cuda":
        return torch.device(f"cuda:{runtime_config.gpu_id}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{runtime_config.gpu_id}")
    return torch.device("cpu")

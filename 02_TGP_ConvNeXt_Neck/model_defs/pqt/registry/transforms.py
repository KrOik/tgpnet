from __future__ import annotations

from .base import Registry

TRANSFORM_REGISTRY: Registry[object] = Registry("transforms")

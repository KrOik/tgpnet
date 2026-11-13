from __future__ import annotations

from pqt.config.schema import ModelConfig
from pqt.model.backbone import BackboneSpec, build_tgp_thermo_seq_v2_backbone


def build_backbone(config: ModelConfig) -> BackboneSpec:
    return build_tgp_thermo_seq_v2_backbone(config)

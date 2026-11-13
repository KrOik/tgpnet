from __future__ import annotations

from pqt.config.schema import ModelConfig
from pqt.experiments.contracts import AssemblyOverrides, ExperimentRuntime

from .assembly import assemble_model_components
from .network import PQTNet


def build_model(
    model_config: ModelConfig,
    assembly_overrides: AssemblyOverrides | None = None,
    experiment: ExperimentRuntime | None = None,
) -> PQTNet:
    if assembly_overrides is not None and experiment is not None:
        raise ValueError("Pass either assembly_overrides or experiment, not both.")
    if experiment is not None:
        assembly_overrides = experiment.assembly_overrides
    components = assemble_model_components(model_config, overrides=assembly_overrides)
    return PQTNet(components)

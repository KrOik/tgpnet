"""Public model imports, resolved lazily to keep experiment/loss imports acyclic."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BackboneSpec": ("pqt.model.backbone", "BackboneSpec"),
    "ModelComponents": ("pqt.model.assembly", "ModelComponents"),
    "PQTNet": ("pqt.model.network", "PQTNet"),
    "PQTRegressionModel": ("pqt.model.network", "PQTRegressionModel"),
    "RegressionOutput": ("pqt.model.outputs", "RegressionOutput"),
    "assemble_model_components": ("pqt.model.assembly", "assemble_model_components"),
    "build_model": ("pqt.model.factory", "build_model"),
    "load_checkpoint": ("pqt.model.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("pqt.model.checkpoint", "save_checkpoint"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

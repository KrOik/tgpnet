from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from pqt.engine.losses import LossBreakdown, compute_loss_breakdown
from pqt.model.outputs import RegressionOutput
from pqt.observability import loss_gradient_attribution


def _torchlens():
    try:
        import torchlens as tl
    except ImportError as exc:
        raise RuntimeError(
            "Model diagnostics require TorchLens. Install it with: python -m pip install -e '.[diagnostics]'"
        ) from exc
    return tl


class DiagnosticAdapter(nn.Module):
    """Tensor-returning wrapper that retains the structured training output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.last_output: RegressionOutput | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = self.model.forward_with_aux(inputs) if hasattr(self.model, "forward_with_aux") else RegressionOutput(self.model(inputs))
        self.last_output = raw
        return raw.prediction


@dataclass(slots=True)
class TracedBatch:
    prediction: torch.Tensor
    output: RegressionOutput
    breakdown: LossBreakdown
    trace: Any
    snapshot_dir: Path
    parameter_before: dict[str, torch.Tensor]


def _scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, float | int | bool]:
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "nan_count": 0, "inf_count": 0}
    detached = tensor.detach().float()
    finite = detached[torch.isfinite(detached)]
    return {
        "mean": float(finite.mean().cpu()) if finite.numel() else math.nan,
        "std": float(finite.std(unbiased=False).cpu()) if finite.numel() else math.nan,
        "min": float(finite.min().cpu()) if finite.numel() else math.nan,
        "max": float(finite.max().cpu()) if finite.numel() else math.nan,
        "nan_count": int(torch.isnan(detached).sum().cpu()),
        "inf_count": int(torch.isinf(detached).sum().cpu()),
    }


def _module_metrics(name: str, module: nn.Module) -> dict[str, Any]:
    parameters = list(module.parameters(recurse=True))
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    weight_sq = sum(float(parameter.detach().float().pow(2).sum().cpu()) for parameter in trainable)
    grads = [parameter.grad for parameter in trainable if parameter.grad is not None]
    grad_sq = sum(float(grad.detach().float().pow(2).sum().cpu()) for grad in grads)
    grad_elements = sum(grad.numel() for grad in grads)
    zero_grad_elements = sum(int((grad.detach() == 0).sum().cpu()) for grad in grads)
    nonfinite_grad_elements = sum(int((~torch.isfinite(grad.detach())).sum().cpu()) for grad in grads)
    return {
        "component": name,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "weight_norm": math.sqrt(weight_sq),
        "grad_norm": math.sqrt(grad_sq),
        "parameters_with_grad": len(grads),
        "parameters_without_grad": len(trainable) - len(grads),
        "zero_grad_fraction": zero_grad_elements / max(1, grad_elements),
        "nonfinite_grad_count": nonfinite_grad_elements,
    }


def _component_modules(adapter: DiagnosticAdapter) -> dict[str, nn.Module]:
    modules = dict(adapter.named_modules())
    components: dict[str, nn.Module] = {
        name: modules[f"model.{name}"]
        for name in ("backbone", "neck", "head", "calibrator")
        if f"model.{name}" in modules
    }
    provider = getattr(adapter.model, "diagnostic_component_paths", None)
    for name, path in (provider() if callable(provider) else {}).items():
        if path in modules:
            components[name] = modules[path]
    return components


def _trace_module_stats(trace: Any) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for record in trace:
        output = getattr(record, "out", None)
        duration = float(getattr(record, "func_duration", 0.0) or 0.0)
        memory = float(getattr(record, "activation_memory", 0.0) or 0.0)
        for address in getattr(record, "output_of_modules", []) or []:
            row = aggregated.setdefault(address, {"forward_seconds": 0.0, "activation_bytes": 0.0})
            row["forward_seconds"] += duration
            row["activation_bytes"] += memory
            if isinstance(output, torch.Tensor):
                stats = _tensor_stats(output)
                for key, value in stats.items():
                    row[f"activation_{key}"] = value
    return aggregated


def _write_trace_artifacts(trace: Any, snapshot_dir: Path) -> None:
    tl = _torchlens()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tl.export.csv(trace, snapshot_dir / "trace.csv")
    tl.export.json(trace, snapshot_dir / "trace.json")
    try:
        trace.draw(
            vis_mode="rolled",
            vis_outpath=str(snapshot_dir / "graph"),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_call_depth=3,
        )
    except Exception as exc:
        (snapshot_dir / "graph_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


def trace_training_batch(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    snapshot_dir: Path,
) -> TracedBatch:
    tl = _torchlens()
    adapter = DiagnosticAdapter(model)
    parameter_before = {
        f"model.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    parameter_before.update(
        {
            f"criterion.{name}": parameter.detach().cpu().clone()
            for name, parameter in criterion.named_parameters()
            if parameter.requires_grad
        }
    )
    trace = tl.trace(
        adapter,
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True, layers_to_save="all"),
        profile=True,
    )
    if adapter.last_output is None:
        raise RuntimeError("Diagnostic adapter did not retain a RegressionOutput.")
    breakdown = compute_loss_breakdown(criterion, adapter.last_output, targets)
    # Capture and export the one managed total-loss backward before the
    # per-term autograd.grad calls below. Otherwise TorchLens records those
    # attribution probes as additional implicit backward passes and its CSV
    # exporter cannot represent a single ``op.grad`` value.
    trace.log_backward(breakdown.total, retain_graph=True)
    _write_trace_artifacts(trace, snapshot_dir)
    attribution_terms = breakdown.weighted_terms or {
        name: value for name, value in breakdown.terms.items() if name != "total" and not name.startswith("weighted_")
    }
    named_parameters = [(f"model.{name}", parameter) for name, parameter in model.named_parameters()]
    named_parameters.extend((f"criterion.{name}", parameter) for name, parameter in criterion.named_parameters())
    loss_gradient_attribution(attribution_terms, named_parameters, snapshot_dir / "loss_gradient_attribution.csv")
    return TracedBatch(adapter.last_output.prediction, adapter.last_output, breakdown, trace, snapshot_dir, parameter_before)


def finalize_training_snapshot(traced: TracedBatch, model: nn.Module, criterion: nn.Module, epoch: int, batch: int) -> None:
    adapter = DiagnosticAdapter(model)
    module_stats = _trace_module_stats(traced.trace)
    rows = []
    for name, module in _component_modules(adapter).items():
        row = {"epoch": epoch, "batch": batch, **_module_metrics(name, module)}
        path = "model." + name if name in {"backbone", "neck", "head", "calibrator"} else None
        provider = getattr(model, "diagnostic_component_paths", None)
        if path is None and callable(provider):
            path = provider().get(name)
        row.update(module_stats.get(path or "", {}))
        rows.append(row)
    rows.append({"epoch": epoch, "batch": batch, **_module_metrics("criterion", criterion)})

    total_grad = sum(
        float(row["grad_norm"])
        for row in rows
        if row["component"] in {"backbone", "neck", "head", "calibrator", "criterion"}
    )
    current_parameters = {f"model.{name}": value for name, value in model.named_parameters()}
    current_parameters.update({f"criterion.{name}": value for name, value in criterion.named_parameters()})
    component_paths = getattr(model, "diagnostic_component_paths", lambda: {})()
    for row in rows:
        if row["component"] in {"backbone", "neck", "head", "calibrator"}:
            prefix = f"model.{row['component']}."
        elif row["component"] == "criterion":
            prefix = "criterion."
        else:
            prefix = component_paths.get(row["component"], "") + "."
        delta_sq = 0.0
        before_sq = 0.0
        for parameter_name, before in traced.parameter_before.items():
            if parameter_name.startswith(prefix) and parameter_name in current_parameters:
                after = current_parameters[parameter_name].detach().cpu()
                delta_sq += float((after - before).float().pow(2).sum())
                before_sq += float(before.float().pow(2).sum())
        row["grad_share"] = float(row["grad_norm"]) / max(total_grad, 1e-30)
        row["update_norm"] = math.sqrt(delta_sq)
        row["update_weight_ratio"] = math.sqrt(delta_sq) / max(math.sqrt(before_sq), 1e-30)
    pd.DataFrame(rows).to_csv(traced.snapshot_dir / "component_summary.csv", index=False)

    loss_rows = [
        {"epoch": epoch, "batch": batch, "kind": "term", "name": name, "value": _scalar(value)}
        for name, value in traced.breakdown.terms.items()
    ] + [
        {"epoch": epoch, "batch": batch, "kind": "state", "name": name, "value": _scalar(value)}
        for name, value in traced.breakdown.state.items()
    ]
    pd.DataFrame(loss_rows).to_csv(traced.snapshot_dir / "loss_terms.csv", index=False)

    auxiliary_rows = []
    routing_anomalies = []
    for name, value in traced.output.auxiliary.items():
        if isinstance(value, torch.Tensor):
            auxiliary_rows.append({"name": name, **_tensor_stats(value)})
            if value.ndim == 2:
                for index, channel in enumerate(value.detach().float().unbind(dim=1)):
                    auxiliary_rows.append({"name": f"{name}[{index}]", **_tensor_stats(channel)})
            if name == "routing_probs" and value.ndim == 2:
                mean_probabilities = value.detach().float().mean(dim=0)
                entropy = -(value.detach().float() * value.detach().float().clamp_min(1e-12).log()).sum(dim=1).mean()
                auxiliary_rows.append({"name": "routing_entropy", **_tensor_stats(entropy.view(1))})
                for index, probability in enumerate(mean_probabilities):
                    if float(probability.cpu()) < 0.01:
                        routing_anomalies.append(
                            {"component": f"head_expert_{index}", "type": "inactive_expert", "mean_probability": float(probability.cpu())}
                        )
                if float(mean_probabilities.max().cpu()) > 0.99:
                    routing_anomalies.append(
                        {"component": "head_router", "type": "routing_collapse", "max_mean_probability": float(mean_probabilities.max().cpu())}
                    )
    pd.DataFrame(auxiliary_rows).to_csv(traced.snapshot_dir / "auxiliary_stats.csv", index=False)

    softmax_rows = []
    for record in traced.trace:
        value = getattr(record, "out", None)
        if getattr(record, "func_name", None) != "softmax" or not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        site = str(getattr(record, "module", None) or "unscoped")
        probabilities = value.detach().float()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1).mean()
        softmax_rows.append({"site": site, "channel": "__entropy__", "mean": float(entropy.cpu())})
        for index, probability in enumerate(probabilities.mean(dim=0)):
            softmax_rows.append({"site": site, "channel": index, "mean": float(probability.cpu())})
    pd.DataFrame(softmax_rows).to_csv(traced.snapshot_dir / "softmax_stats.csv", index=False)

    anomalies = list(routing_anomalies)
    for row in rows:
        if row["nonfinite_grad_count"]:
            anomalies.append({"component": row["component"], "type": "nonfinite_gradient", "count": row["nonfinite_grad_count"]})
        if row["trainable_parameter_count"] and row["parameters_with_grad"] == 0:
            anomalies.append({"component": row["component"], "type": "missing_gradient"})
        elif row["parameters_with_grad"] and row["grad_norm"] == 0:
            anomalies.append({"component": row["component"], "type": "zero_gradient"})
    (traced.snapshot_dir / "anomalies.json").write_text(json.dumps(anomalies, indent=2) + "\n", encoding="utf-8")
    traced.trace.cleanup()


def run_causal_ablations(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_dir: Path,
) -> Path:
    tl = _torchlens()
    adapter = DiagnosticAdapter(model)
    with torch.no_grad():
        baseline_prediction = adapter(inputs)
        if adapter.last_output is None:
            raise RuntimeError("Diagnostic adapter did not retain a baseline output.")
        baseline_loss = _scalar(compute_loss_breakdown(criterion, adapter.last_output, targets).total)
        baseline_rmse = float(torch.sqrt(torch.mean((baseline_prediction - targets) ** 2)).cpu())
        baseline_mae = float(torch.mean(torch.abs(baseline_prediction - targets)).cpu())

    rows = [{"component": "__baseline__", "loss": baseline_loss, "loss_delta": 0.0, "rmse": baseline_rmse, "rmse_delta": 0.0, "mae": baseline_mae, "mae_delta": 0.0}]
    provider = getattr(model, "diagnostic_component_paths", None)
    for component, path in (provider() if callable(provider) else {}).items():
        trace = tl.trace(
            adapter,
            inputs,
            intervene=tl.when(tl.module(path), tl.zero_ablate()),
            capture=tl.options.CaptureOptions(inference_only=True, layers_to_save="none"),
        )
        if adapter.last_output is None:
            trace.cleanup()
            continue
        prediction = adapter.last_output.prediction
        loss = _scalar(compute_loss_breakdown(criterion, adapter.last_output, targets).total)
        rmse = float(torch.sqrt(torch.mean((prediction - targets) ** 2)).detach().cpu())
        mae = float(torch.mean(torch.abs(prediction - targets)).detach().cpu())
        rows.append({"component": component, "module_path": path, "loss": loss, "loss_delta": loss - baseline_loss, "rmse": rmse, "rmse_delta": rmse - baseline_rmse, "mae": mae, "mae_delta": mae - baseline_mae})
        trace.cleanup()
    output_path = output_dir / "ablation_contributions.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

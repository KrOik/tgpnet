from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from pqt.config.schema import DiagnosticsConfig
from pqt.engine.losses import LossBreakdown
from pqt.model.outputs import RegressionOutput


def _finite_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().mean().cpu())
    return float(value)


class OnlineMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, count: int = 1) -> None:
        if math.isfinite(value):
            self.total += value * count
            self.count += count

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else math.nan


@dataclass(slots=True)
class Event:
    run: str
    split: str
    epoch: int
    step: int
    phase: str
    scope: str
    name: str
    value: float


class TelemetryObserver:
    """Online scalar aggregation. No batch tensor is retained."""

    def __init__(self) -> None:
        self.metrics: dict[str, OnlineMean] = defaultdict(OnlineMean)

    def reset(self) -> None:
        self.metrics = defaultdict(OnlineMean)

    def tensor(self, name: str, value: torch.Tensor) -> None:
        data = value.detach().float()
        if not data.numel():
            return
        self.metrics[f"tensor.{name}.mean"].add(float(data.mean().cpu()))
        self.metrics[f"tensor.{name}.std"].add(float(data.std(unbiased=False).cpu()))
        self.metrics[f"tensor.{name}.near_zero_fraction"].add(float((data.abs() < 1e-6).float().mean().cpu()))
        if data.ndim == 2 and data.shape[1] > 1:
            probabilities = data.clamp_min(1e-12)
            if torch.allclose(data.sum(1), torch.ones_like(data[:, 0]), atol=1e-3):
                entropy = -(probabilities * probabilities.log()).sum(1)
                self.metrics[f"tensor.{name}.entropy"].add(float(entropy.mean().cpu()))
                self.metrics[f"tensor.{name}.effective_count"].add(float(entropy.exp().mean().cpu()))
                self.metrics[f"tensor.{name}.max"].add(float(data.max(1).values.mean().cpu()))
                usage = torch.bincount(data.argmax(1), minlength=data.shape[1]).float() / data.shape[0]
                for index, item in enumerate(data.mean(0)):
                    self.metrics[f"tensor.{name}.channel_{index}.mean"].add(float(item.cpu()))
                    self.metrics[f"tensor.{name}.channel_{index}.top1_usage"].add(float(usage[index].cpu()))

    def loss(self, breakdown: LossBreakdown) -> None:
        source = breakdown.weighted_terms or breakdown.terms
        total = abs(_finite_scalar(breakdown.total))
        for name, value in source.items():
            if name == "total":
                continue
            scalar = _finite_scalar(value)
            self.metrics[f"loss.{name}.value"].add(scalar)
            self.metrics[f"loss.{name}.absolute_share"].add(abs(scalar) / max(total, 1e-30))
        self.metrics["loss.total.value"].add(_finite_scalar(breakdown.total))
        for name, value in breakdown.state.items():
            self.metrics[f"loss.state.{name}"].add(_finite_scalar(value))

    def output(self, output: RegressionOutput, targets: torch.Tensor) -> None:
        for name, value in output.auxiliary.items():
            if isinstance(value, torch.Tensor):
                self.tensor(name, value)
        confidence = output.auxiliary.get("confidence")
        if isinstance(confidence, torch.Tensor):
            confidence = torch.sigmoid(confidence.detach().float().view(-1))
            error = (output.prediction.detach().float().view(-1) - targets.detach().float().view(-1)).abs()
            self.metrics["confidence.mean"].add(float(confidence.mean().cpu()))
            self.metrics["confidence.saturation_fraction"].add(float(((confidence < .01) | (confidence > .99)).float().mean().cpu()))
            if confidence.numel() > 1 and confidence.std(unbiased=False) > 0 and error.std(unbiased=False) > 0:
                corr = torch.corrcoef(torch.stack((confidence, error)))[0, 1]
                self.metrics["confidence.error_correlation"].add(float(corr.cpu()))

    def values(self) -> dict[str, float]:
        return {name: aggregate.value for name, aggregate in self.metrics.items()}


class OptimizationObserver:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.before: dict[int, torch.Tensor] = {}

    def batch_forward(self) -> None:
        self.before = {id(p): p.detach().clone() for group in self.optimizer.param_groups for p in group["params"]}

    def backward_metrics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        norms: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("name", f"group_{index}"))
            parameters = [p for p in group["params"] if p.requires_grad]
            grads = [p.grad for p in parameters if p.grad is not None]
            norms[name] = math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads))
            result[f"optimization.{name}.grad_norm"] = norms[name]
            result[f"optimization.{name}.missing_grad_fraction"] = (len(parameters) - len(grads)) / max(1, len(parameters))
            result[f"optimization.{name}.nonfinite_grad_count"] = float(sum(int((~torch.isfinite(g)).sum().cpu()) for g in grads))
        total = sum(norms.values())
        for name, norm in norms.items():
            result[f"optimization.{name}.grad_share"] = norm / max(total, 1e-30)
        return result

    def optimizer_metrics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("name", f"group_{index}"))
            params = list(group["params"])
            weight_sq = sum(float(p.detach().float().square().sum().cpu()) for p in params)
            update_sq = sum(float((p.detach() - self.before[id(p)]).float().square().sum().cpu()) for p in params if id(p) in self.before)
            result[f"optimization.{name}.weight_norm"] = math.sqrt(weight_sq)
            result[f"optimization.{name}.update_norm"] = math.sqrt(update_sq)
            result[f"optimization.{name}.update_weight_ratio"] = math.sqrt(update_sq) / max(math.sqrt(weight_sq), 1e-30)
            result[f"lr.{name}"] = float(group["lr"])
        self.before.clear()
        return result


class FindingObserver:
    def __init__(self, config: DiagnosticsConfig) -> None:
        self.config = config
        self.streaks: dict[str, int] = defaultdict(int)
        self.findings: list[dict[str, Any]] = []

    def consume(self, epoch: int, metrics: dict[str, float]) -> list[dict[str, Any]]:
        candidates: list[tuple[str, str, float, float, str]] = []
        for path, value in metrics.items():
            if path.endswith("nonfinite_grad_count") and value > 0:
                candidates.append(("nonfinite_gradient", path, value, 0.0, "critical"))
            elif path.endswith("missing_grad_fraction") and value >= 1:
                candidates.append(("missing_gradient", path, value, 1.0, "critical"))
            elif path.endswith("update_weight_ratio") and value <= self.config.near_zero_threshold:
                candidates.append(("component_not_learning", path, value, self.config.near_zero_threshold, "warning"))
            elif path.endswith("effective_count") and value < self.config.effective_count_threshold:
                kind = "expert_collapse" if "routing" in path else "scale_collapse"
                candidates.append((kind, path, value, self.config.effective_count_threshold, "warning"))
            elif path.endswith(".max") and value > self.config.dominance_threshold:
                candidates.append(("routing_or_scale_dominance", path, value, self.config.dominance_threshold, "warning"))
            elif path.endswith("absolute_share") and value > self.config.dominance_threshold:
                candidates.append(("loss_dominance", path, value, self.config.dominance_threshold, "warning"))
            elif path == "confidence.saturation_fraction" and value > self.config.dominance_threshold:
                candidates.append(("confidence_saturation", path, value, self.config.dominance_threshold, "warning"))
        active = {f"{kind}:{path}" for kind, path, *_ in candidates}
        for key in list(self.streaks):
            if key not in active:
                self.streaks[key] = 0
        emitted = []
        for kind, path, value, threshold, severity in candidates:
            key = f"{kind}:{path}"
            self.streaks[key] += 1
            immediate = severity == "critical"
            if (immediate or self.streaks[key] >= self.config.collapse_patience) and not any(f["key"] == key for f in self.findings):
                finding = {"key": key, "type": kind, "severity": severity, "epoch": epoch, "evidence": {"metric_path": path, "value": value, "threshold": threshold}, "suggestion": "Inspect the named component and the corresponding snapshot."}
                self.findings.append(finding)
                emitted.append(finding)
        return emitted


class ObserverHub:
    phases = ("train_start", "batch_forward", "loss_ready", "backward_done", "optimizer_done", "epoch_end", "train_end")

    def __init__(self, output_dir: Path, config: DiagnosticsConfig, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, run_id: str = "training") -> None:
        self.root = output_dir / config.output_subdir
        self.root.mkdir(parents=True, exist_ok=True)
        self.config, self.run_id = config, run_id
        self.telemetry = TelemetryObserver()
        self.optimization = OptimizationObserver(model, criterion, optimizer)
        self.finder = FindingObserver(config)
        self.events_path = self.root / "events.jsonl"
        self.summary_path = self.root / "epoch_summary.csv"
        self.summary_rows: list[dict[str, float | int]] = []
        self.pending: dict[str, OnlineMean] = defaultdict(OnlineMean)
        self.anomaly_triggered = False

    def _add(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.pending[name].add(value)

    def train_start(self) -> None:
        self.events_path.write_text("", encoding="utf-8")

    def sink(self, name: str, value: torch.Tensor) -> None:
        self.telemetry.tensor(name, value)

    def batch_forward(
        self,
        output: RegressionOutput,
        targets: torch.Tensor,
        *,
        track_optimization: bool = True,
    ) -> None:
        if track_optimization:
            self.optimization.batch_forward()
        self.telemetry.output(output, targets)

    def loss_ready(self, breakdown: LossBreakdown) -> None:
        self.telemetry.loss(breakdown)

    def backward_done(self) -> None:
        self._add(self.optimization.backward_metrics())

    def optimizer_done(self) -> None:
        self._add(self.optimization.optimizer_metrics())

    def should_snapshot(self, epoch: int, batch: int) -> bool:
        scheduled = self.config.enabled and ((self.config.snapshot_first_epoch and epoch == 1) or epoch % self.config.snapshot_interval == 0)
        triggered = self.config.enabled and self.config.trigger_on_anomaly and self.anomaly_triggered
        return (scheduled or triggered) and batch <= self.config.batches_per_snapshot

    def epoch_end(self, epoch: int) -> list[dict[str, Any]]:
        metrics = self.telemetry.values()
        metrics.update({name: value.value for name, value in self.pending.items()})
        row: dict[str, float | int] = {"epoch": epoch, **metrics}
        self.summary_rows.append(row)
        findings = self.finder.consume(epoch, metrics)
        self.anomaly_triggered = bool(findings)
        with self.events_path.open("a", encoding="utf-8") as handle:
            for name, value in metrics.items():
                scope, _, metric = name.partition(".")
                event = Event(self.run_id, "train", epoch, 0, "epoch_end", scope, metric or scope, value)
                handle.write(json.dumps(event.__dict__ if hasattr(event, "__dict__") else {field: getattr(event, field) for field in Event.__dataclass_fields__}, allow_nan=True) + "\n")
        fieldnames = sorted({key for item in self.summary_rows for key in item}, key=lambda key: (key != "epoch", key))
        with self.summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(self.summary_rows)
        self.telemetry.reset(); self.pending = defaultdict(OnlineMean)
        return findings

    def train_end(self) -> dict[str, str]:
        findings_path = self.root / "findings.json"
        findings_path.write_text(json.dumps(self.finder.findings, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        markdown = ["# Training findings", ""]
        if not self.finder.findings:
            markdown.append("No threshold-backed findings were detected.")
        for finding in self.finder.findings:
            evidence = finding["evidence"]
            markdown.extend([f"## {finding['type']} ({finding['severity']})", "", f"Epoch {finding['epoch']}; `{evidence['metric_path']}` = {evidence['value']:.6g} (threshold {evidence['threshold']:.6g}).", "", finding["suggestion"], ""])
        findings_md = self.root / "findings.md"
        findings_md.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
        return {"events": str(self.events_path), "epoch_summary": str(self.summary_path), "findings": str(findings_path), "findings_markdown": str(findings_md), "snapshots": str(self.root / "snapshots")}


def loss_gradient_attribution(terms: dict[str, torch.Tensor], named_parameters: Iterable[tuple[str, nn.Parameter]], output_path: Path) -> Path:
    """Measure per-loss gradients with autograd.grad; existing .grad buffers are untouched."""
    parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    rows: list[dict[str, Any]] = []
    vectors: dict[str, torch.Tensor] = {}
    active = [(name, term) for name, term in terms.items() if name != "total" and term.requires_grad]
    for index, (term_name, term) in enumerate(active):
        grads = torch.autograd.grad(term, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        vector = torch.cat([(g.detach().reshape(-1).cpu() if g is not None else torch.zeros(p.numel())) for (_, p), g in zip(parameters, grads, strict=True)])
        vectors[term_name] = vector
        rows.append({"kind": "norm", "loss": term_name, "component": "all", "value": float(vector.norm())})
    for left_index, (left, left_vector) in enumerate(vectors.items()):
        for right, right_vector in list(vectors.items())[left_index + 1:]:
            cosine = float(torch.dot(left_vector, right_vector) / (left_vector.norm() * right_vector.norm()).clamp_min(1e-30))
            rows.append({"kind": "cosine", "loss": left, "component": right, "value": cosine})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("kind", "loss", "component", "value")); writer.writeheader(); writer.writerows(rows)
    return output_path

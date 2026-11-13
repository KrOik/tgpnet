from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from pqt.engine.losses import LossBreakdown
from pqt.depth_ordinal import ordinal_targets
from pqt.model.outputs import RegressionOutput, as_regression_output

EPSILON = 1e-6


def _compute_label_stats(train_config):
    frame = pd.read_csv(train_config.data.csv_path)
    labels = frame[train_config.data.label_column].astype(float)
    quantiles = labels.quantile([0.2, 0.35, 0.5, 0.65, 0.8]).to_dict()
    std_value = float(labels.std()) or 1.0
    return {
        "q20": float(quantiles[0.2]), "q35": float(quantiles[0.35]),
        "q50": float(quantiles[0.5]), "q65": float(quantiles[0.65]),
        "q80": float(quantiles[0.8]), "mean": float(labels.mean()),
        "std": std_value,
        "iqr": float(labels.quantile(0.8) - labels.quantile(0.2)) or std_value,
    }


class DepthRegressionLoss(nn.Module):
    """Canonical and candidate objectives with normalized depth weighting."""

    def __init__(self, *, loss_name: str, median_value: float, iqr_value: float,
                 huber_delta: float, quantile_tau: float, depth_weight_gamma: float,
                 depth_max: float | None, ordinal_loss_weight: float = 0.0) -> None:
        super().__init__()
        self.loss_name = loss_name
        self.median_value = median_value
        self.iqr_value = iqr_value
        self.huber_delta = huber_delta
        self.quantile_tau = quantile_tau
        self.depth_weight_gamma = depth_weight_gamma
        self.depth_max = depth_max
        self.ordinal_loss_weight = ordinal_loss_weight
        if loss_name in {"structured_canonical", "canonical"}:
            # Names and initialization intentionally match the v0.3.0 criterion state.
            self.raw_alpha = nn.Parameter(torch.tensor(1.0))
            self.raw_scale = nn.Parameter(torch.tensor(1.0))
        if depth_weight_gamma > 0 and (depth_max is None or depth_max <= 0):
            raise ValueError("Positive depth weighting requires depth_max from the current training fold.")

    def _barron_values(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.raw_alpha) * (2.0 - 2.0 * EPSILON) + EPSILON
        scale = F.softplus(self.raw_scale) + EPSILON
        squared = ((prediction - target) / scale).pow(2)
        beta = torch.clamp(torch.abs(alpha - 2.0), min=EPSILON)
        generic = (beta / alpha) * (torch.pow(squared / beta + 1.0, alpha / 2.0) - 1.0)
        return torch.where(alpha < 1e-3, torch.log1p(0.5 * squared), generic)

    def _per_sample(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name in {"structured_canonical", "canonical"}:
            centered = target - self.median_value
            transition = torch.exp(-torch.square(centered / max(0.5 * self.iqr_value, EPSILON)))
            return self._barron_values(prediction, target) * (1.0 + 0.35 * transition)
        if self.loss_name == "mse":
            return torch.square(prediction - target)
        if self.loss_name == "huber":
            return F.huber_loss(prediction, target, reduction="none", delta=self.huber_delta)
        if self.loss_name == "quantile":
            residual = target - prediction
            return torch.maximum(self.quantile_tau * residual, (self.quantile_tau - 1.0) * residual)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def depth_weights(self, target: torch.Tensor) -> torch.Tensor:
        if self.depth_weight_gamma == 0:
            return torch.ones_like(target)
        return 1.0 + self.depth_weight_gamma * target / float(self.depth_max)

    def _main_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values = self._per_sample(prediction, target)
        weights = self.depth_weights(target)
        return torch.sum(weights * values) / torch.sum(weights)

    def forward(self, predictions: torch.Tensor | RegressionOutput, targets: torch.Tensor) -> torch.Tensor:
        return self.forward_with_breakdown(predictions, targets).total

    def forward_with_breakdown(self, predictions: torch.Tensor | RegressionOutput,
                               targets: torch.Tensor) -> LossBreakdown:
        output = as_regression_output(predictions)
        target = targets.view(-1)
        prediction = output.prediction.view(-1)
        main = self._main_loss(prediction, target)
        main_name = "main_barron" if self.loss_name in {"structured_canonical", "canonical"} else self.loss_name
        terms = {main_name: main}
        raw_terms = {main_name: main}
        total = main

        ordinal_logits = output.auxiliary.get("ordinal_logits")
        if ordinal_logits is not None:
            ordinal = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets(target))
            raw_terms["ordinal"] = ordinal
            terms["ordinal"] = self.ordinal_loss_weight * ordinal
            total = total + terms["ordinal"]

        if self.loss_name == "structured_canonical":
            payload = output.auxiliary
            zero = main.new_zeros(())
            terms.update(routing_entropy=zero, routing_balance=zero, confidence=zero,
                         expert_collapse=zero, routing_peak=zero)
            routing_probs = payload.get("routing_probs")
            confidence = payload.get("confidence")
            if routing_probs is not None:
                entropy = -(routing_probs * torch.log(routing_probs.clamp_min(EPSILON))).sum(dim=1).mean()
                balance = torch.square(
                    routing_probs.mean(dim=0)
                    - torch.full_like(routing_probs.mean(dim=0), 1.0 / routing_probs.shape[1])
                ).mean()
                raw_terms.update(routing_entropy=entropy, routing_balance=balance)
                terms["routing_entropy"] = 0.020 * entropy
                terms["routing_balance"] = 0.030 * balance
                total = total + terms["routing_entropy"] + terms["routing_balance"]
            if confidence is not None:
                detached_error = torch.abs(prediction.detach() - target)
                target_confidence = torch.exp(-detached_error / max(0.5 * self.iqr_value, EPSILON))
                raw_terms["confidence"] = F.mse_loss(torch.sigmoid(confidence.view(-1)), target_confidence)
                terms["confidence"] = 0.040 * raw_terms["confidence"]
                total = total + terms["confidence"]
            expert_outputs = payload.get("expert_outputs")
            if expert_outputs is not None and routing_probs is not None:
                collapse = F.relu(0.050 - expert_outputs.std(dim=1)).mean()
                peak = (1.0 - routing_probs.max(dim=1).values).mean()
                raw_terms.update(expert_collapse=collapse, routing_peak=peak)
                terms["expert_collapse"] = 0.020 * collapse
                terms["routing_peak"] = 0.015 * peak
                total = total + terms["expert_collapse"] + terms["routing_peak"]

        terms["total"] = total
        state: dict[str, torch.Tensor] = {"depth_weight_gamma": main.new_tensor(self.depth_weight_gamma)}
        if self.depth_max is not None:
            state["depth_max"] = main.new_tensor(self.depth_max)
        if self.loss_name in {"structured_canonical", "canonical"}:
            state.update(
                alpha=torch.sigmoid(self.raw_alpha) * (2.0 - 2.0 * EPSILON) + EPSILON,
                scale=F.softplus(self.raw_scale) + EPSILON,
            )
        return LossBreakdown(total=total, raw_terms=raw_terms,
                             weighted_terms={name: value for name, value in terms.items() if name != "total"},
                             terms=terms, state=state)


def build_loss(train_config):
    stats = _compute_label_stats(train_config)
    optimization = train_config.optimization
    if train_config.model.head_variant == "normalized_four_scale":
        if optimization.loss_name != "huber" or abs(optimization.huber_delta - 0.08) > 1e-12:
            raise ValueError("normalized_four_scale requires Huber loss with delta=0.08.")
        if optimization.ordinal_loss_weight not in {0.0, 0.001}:
            raise ValueError("normalized_four_scale ordinal_loss_weight must be 0 or 0.001.")
    return DepthRegressionLoss(
        loss_name=optimization.loss_name,
        median_value=float(stats["q50"]), iqr_value=float(stats["iqr"]),
        huber_delta=optimization.huber_delta, quantile_tau=optimization.quantile_tau,
        depth_weight_gamma=optimization.depth_weight_gamma, depth_max=optimization.depth_max,
        ordinal_loss_weight=(
            optimization.ordinal_loss_weight
            if train_config.model.head_variant in {"ordinal_residual", "normalized_four_scale"}
            else 0.0
        ),
    )

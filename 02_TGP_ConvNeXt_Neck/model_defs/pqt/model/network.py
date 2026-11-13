from __future__ import annotations

import torch
from torch import nn

from .assembly import ModelComponents
from .backbone import BackboneSpec
from .outputs import RegressionOutput, as_regression_output
from .inputs import DualResolutionInput, primary_model_input


class PQTNet(nn.Module):
    def __init__(
        self,
        components: ModelComponents,
    ) -> None:
        super().__init__()
        self.backbone = components.backbone.module
        self.extract_backbone_features = components.backbone.extract_features
        self.feature_dim = components.backbone.feature_dim
        capture_indices = tuple(getattr(self.backbone, "capture_indices", ()))
        self.feature_taps = nn.ModuleList(nn.Identity() for _ in capture_indices)
        self.neck = components.neck.module
        self.embedding_dim = components.neck.output_dim
        self.head = components.head
        self.calibrator = components.calibrator
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

    def forward_with_aux(self, x: torch.Tensor | DualResolutionInput) -> RegressionOutput:
        backbone_input = x if getattr(self.backbone, "expects_dual_resolution", False) else primary_model_input(x)
        features = self.extract_backbone_features(self.backbone, backbone_input)
        if self.feature_taps and isinstance(features, (list, tuple)):
            if len(features) != len(self.feature_taps):
                raise RuntimeError(f"Expected {len(self.feature_taps)} tapped features, got {len(features)}.")
            features = [tap(feature) for tap, feature in zip(self.feature_taps, features, strict=True)]
        features = self.neck(features)
        output = as_regression_output(self.head(features))
        return RegressionOutput(
            prediction=self.calibrator(output.prediction),
            auxiliary=output.auxiliary,
        )

    def forward(self, x: torch.Tensor | DualResolutionInput) -> torch.Tensor:
        return self.forward_with_aux(x).prediction

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        if hasattr(self.backbone, "parameter_groups"):
            backbone_groups = self.backbone.parameter_groups()
        else:
            backbone_groups = {
                "backbone": [
                    parameter for name, parameter in self.backbone.named_parameters()
                    if parameter.requires_grad and not name.startswith("classifier.")
                ]
            }
        return {
            **backbone_groups,
            "neck": [parameter for parameter in self.neck.parameters() if parameter.requires_grad],
            "head": [parameter for parameter in self.head.parameters() if parameter.requires_grad],
            "calibrator": [parameter for parameter in self.calibrator.parameters() if parameter.requires_grad],
        }

    def diagnostic_component_paths(self) -> dict[str, str]:
        """Stable semantic sites used for component-level trace and intervention."""
        paths: dict[str, str] = {}
        if hasattr(self.backbone, "diagnostic_component_paths"):
            paths.update(self.backbone.diagnostic_component_paths())
        for index in range(len(self.feature_taps)):
            paths[f"backbone_scale_{index}"] = f"model.feature_taps.{index}"
        for prefix, module in (("neck", self.neck), ("head", self.head)):
            for child_name, child in module.named_children():
                if isinstance(child, nn.ModuleList):
                    for index in range(len(child)):
                        paths[f"{prefix}_{child_name}_{index}"] = f"model.{prefix}.{child_name}.{index}"
                else:
                    paths[f"{prefix}_{child_name}"] = f"model.{prefix}.{child_name}"
        return paths


PQTRegressionModel = PQTNet

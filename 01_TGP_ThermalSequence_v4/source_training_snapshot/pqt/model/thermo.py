from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .inputs import NeighborhoodInput


class StripeImageToCurve(nn.Module):
    """Collapse a row-replicated stripe image into its 1-D thermal curve.

    The manifest supplies 1024x1024 grayscale PNGs whose rows are identical:
    the first row already encodes the 1024-sample response of one pixel.
    Averaging over rows is an identity under that contract and remains correct
    if minor row noise is ever present. The output is denormalized to [0, 1].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"StripeImageToCurve expects single-channel NCHW input, got {tuple(x.shape)}")
        curve = x.mean(dim=(1, 2))
        return (curve.clamp(-1.0, 1.0) + 1.0) * 0.5


def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(4.0 * sigma)))
    grid = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(grid * grid) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


class MultiScaleDerivativeEncoding(nn.Module):
    """Scale-normalized smoothed-derivative encoding of a thermal curve.

    Emits ``1 + 3 * len(scales)`` channels: the raw curve and, per scale
    ``sigma``, the Gaussian-smoothed curve ``u_sigma``, its scale-normalized
    first derivative ``sigma * Du_sigma`` and second derivative
    ``sigma^2 * D^2 u_sigma``. Smoothing suppresses the 8-bit quantization
    noise that finite differences would otherwise amplify.
    """

    def __init__(
        self,
        scales: tuple[float, ...],
        enabled: bool = True,
        include_raw: bool = True,
    ) -> None:
        super().__init__()
        if enabled and not scales:
            raise ValueError("MultiScaleDerivativeEncoding requires at least one scale.")
        self.enabled = enabled
        self.include_raw = include_raw
        self.scales = tuple(float(scale) for scale in scales)
        for index, sigma in enumerate(self.scales):
            if sigma <= 0:
                raise ValueError(f"Encoding scales must be positive, got {sigma}")
            self.register_buffer(f"kernel_{index}", _gaussian_kernel1d(sigma).view(1, 1, -1))
        self.out_channels = (1 if include_raw else 0) + 3 * len(self.scales) if enabled else 1

    @staticmethod
    def _smooth(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if kernel.shape[-1] > length:
            # A fixed-radius kernel must not dominate a short sequence: crop to
            # the largest odd window <= L and renormalize, capping the
            # replicate-padded smear region at L/4 as the reviewer requested.
            target = min(2 * (length // 4) + 1, kernel.shape[-1])
            start = (kernel.shape[-1] - target) // 2
            kernel = kernel[..., start : start + target] / kernel[..., start : start + target].sum()
        pad = kernel.shape[-1] // 2
        return F.conv1d(F.pad(x, (pad, pad), mode="replicate"), kernel)

    @staticmethod
    def _difference(x: torch.Tensor) -> torch.Tensor:
        return torch.diff(x, dim=-1, prepend=x[..., :1])

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        if curve.ndim != 2:
            raise ValueError(f"MultiScaleDerivativeEncoding expects [B, L], got {tuple(curve.shape)}")
        signal = curve.unsqueeze(1)
        if not self.enabled:
            return signal
        channels = [signal] if self.include_raw else []
        for index, sigma in enumerate(self.scales):
            smoothed = self._smooth(signal, getattr(self, f"kernel_{index}"))
            first = self._difference(smoothed) * sigma
            second = self._difference(first) * sigma
            channels.extend((smoothed, first, second))
        return torch.cat(channels, dim=1)


class MultiScaleKernelBlock(nn.Module):
    """Residual block mixing depthwise temporal convolutions of several widths."""

    def __init__(self, channels: int, kernel_sizes: tuple[int, ...] = (3, 9, 27), expansion: int = 4) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("MultiScaleKernelBlock requires at least one kernel size.")
        self.branches = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=size, padding=size // 2, groups=channels)
            for size in kernel_sizes
        )
        self.mix = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels * expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels * expansion, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mix(torch.cat([branch(x) for branch in self.branches], dim=1))
        x = x + mixed
        return x + self.mlp(self.norm(x))


class TemporalStageEncoder(nn.Module):
    """Stem plus four residual stages; returns one feature map per stage.

    Total stride is 2 * 2^3 = 16, so a 1024-sample curve produces stage
    lengths of roughly 512/256/128/64.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        blocks_per_stage: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 9, 27),
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("TemporalStageEncoder requires at least two stages.")
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(1, channels[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*(MultiScaleKernelBlock(width, kernel_sizes) for _ in range(blocks_per_stage)))
            for width in channels
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels[index], channels[index + 1], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, channels[index + 1]),
            )
            for index in range(len(channels) - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return outputs


class TemporalPositionDescriptor(nn.Module):
    """Fixed 'when does the response change' features computed from the curve.

    All features are positions normalized to [0, 1], pooled derivative
    statistics, or level summaries of the smoothed curve. They are computed
    under ``no_grad`` and act as conditioning inputs to the projection layer;
    the descriptor itself has no trainable parameters.
    """

    window_count = 8
    out_features = 16

    def __init__(self, smooth_sigma: float = 8.0, soft: bool = False) -> None:
        super().__init__()
        if smooth_sigma <= 0:
            raise ValueError(f"smooth_sigma must be positive, got {smooth_sigma}")
        self.soft = bool(soft)
        self.register_buffer("kernel", _gaussian_kernel1d(smooth_sigma).view(1, 1, -1))

    def _positions(self, derivative: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.soft:
            return (
                derivative.argmax(dim=-1).float() / length,
                derivative.argmin(dim=-1).float() / length,
            )
        # Softargmax is continuous in the input; hard argmax positions are not,
        # which amplified small curve perturbations into large feature jumps.
        scale = derivative.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        grid = torch.linspace(0.0, 1.0, length, device=derivative.device, dtype=derivative.dtype)
        pos_rise = (F.softmax(derivative / scale, dim=-1) * grid).sum(dim=-1)
        pos_fall = (F.softmax(-derivative / scale, dim=-1) * grid).sum(dim=-1)
        return pos_rise, pos_fall

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            signal = curve.unsqueeze(1)
            smoothed = MultiScaleDerivativeEncoding._smooth(signal, self.kernel).squeeze(1)
            derivative = torch.diff(smoothed, dim=-1, prepend=smoothed[..., :1])
            length = smoothed.shape[-1]
            edge = max(1, length // self.window_count)

            amplitude = smoothed.amax(dim=-1) - smoothed.amin(dim=-1)
            pos_rise, pos_fall = self._positions(derivative, length)
            magnitude = derivative.abs()
            total = magnitude.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            cumulative = magnitude.cumsum(dim=-1) / total
            quantile_positions = torch.stack(
                [(cumulative < fraction).sum(dim=-1).float() / length for fraction in (0.25, 0.5, 0.75)],
                dim=-1,
            )
            windowed = F.adaptive_avg_pool1d(derivative.unsqueeze(1), self.window_count).squeeze(1) * length
            early = smoothed[..., :edge].mean(dim=-1)
            late = smoothed[..., -edge:].mean(dim=-1)

            return torch.cat(
                (
                    amplitude.unsqueeze(1),
                    pos_rise.unsqueeze(1),
                    pos_fall.unsqueeze(1),
                    quantile_positions,
                    windowed,
                    early.unsqueeze(1),
                    late.unsqueeze(1),
                ),
                dim=-1,
            )


class ThermalGlobalStats(nn.Module):
    """Amplitude-preserving global statistics of the response curve.

    LayerNorm inside the projection strips absolute scale, which is the
    primary depth cue in pulsed thermography. These raw statistics re-inject
    it downstream of the pooled features. Computed under ``no_grad``.
    """

    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    out_features = 12

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = torch.quantile(
                curve, curve.new_tensor(self.quantiles), dim=-1
            ).transpose(0, 1)
            amplitude = curve.amax(dim=-1) - curve.amin(dim=-1)
            edge = max(1, curve.shape[-1] // 8)
            return torch.cat(
                (
                    curve.mean(dim=-1, keepdim=True),
                    curve.std(dim=-1, keepdim=True),
                    curve.amin(dim=-1, keepdim=True),
                    curve.amax(dim=-1, keepdim=True),
                    torch.log(amplitude.clamp_min(1e-6)).unsqueeze(1),
                    q,
                    curve[..., :edge].mean(dim=-1, keepdim=True),
                    curve[..., -edge:].mean(dim=-1, keepdim=True),
                ),
                dim=-1,
            )


class ThermalSpectralBranch(nn.Module):
    """Low-frequency rFFT descriptor: log-magnitude and phase per bin.

    Depth in pulsed thermography manifests as a diffusion low-pass over the
    stimulus, so the spectral envelope carries information orthogonal to the
    temporal CNN. Emits ``out_dim`` features from a small MLP.
    """

    def __init__(self, bins: int, out_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        if bins < 1:
            raise ValueError("spectral bins must be positive.")
        self.bins = int(bins)
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(curve, dim=-1)[..., : self.bins]
        magnitude = spectrum.abs()
        phase = torch.atan2(spectrum.imag, spectrum.real)
        features = torch.cat(
            (
                torch.log(magnitude + 1e-8),
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )
        pad = 3 * self.bins - features.shape[-1]
        if pad > 0:
            features = F.pad(features, (0, pad))
        return self.mlp(features)


class TGPThermoSeqBackbone(nn.Module):
    """Custom 1-D backbone for row-replicated thermal stripe images.

    Pipeline: stripe image -> response curve -> multi-scale derivative
    encoding -> four residual temporal stages -> segmented time-position
    pooling per stage + handcrafted position descriptors -> projection to a
    ``embed_dim`` descriptor. The downstream contract is a plain [B, D]
    tensor, so pair it with the ``identity`` neck.
    """

    expects_dual_resolution = False

    def __init__(
        self,
        *,
        scales: tuple[float, ...] = (4.0, 16.0, 64.0),
        channels: tuple[int, ...] = (32, 64, 128, 192),
        blocks_per_stage: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 9, 27),
        segments: tuple[int, ...] = (1, 2, 4),
        embed_dim: int = 256,
        multiscale_encoding: bool = True,
        include_raw: bool = True,
        position_features: bool = True,
        soft_position: bool = False,
        global_stats: bool = False,
        spectral_bins: int = 0,
        spectral_embed_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.curve_extractor = StripeImageToCurve()
        self.encoding = MultiScaleDerivativeEncoding(
            scales, enabled=multiscale_encoding, include_raw=include_raw
        )
        self.encoder = TemporalStageEncoder(self.encoding.out_channels, channels, blocks_per_stage, kernel_sizes)
        self.segments = tuple(int(segment) for segment in segments)
        if not self.segments or any(segment < 1 for segment in self.segments):
            raise ValueError(f"segments must be positive integers, got {self.segments}")
        pooled_width = sum(self.segments)
        self.stage_norms = nn.ModuleList(nn.LayerNorm(width * pooled_width) for width in channels)
        self.position = TemporalPositionDescriptor(soft=soft_position) if position_features else None
        self.stats = ThermalGlobalStats() if global_stats else None
        self.spectral = ThermalSpectralBranch(spectral_bins, out_dim=spectral_embed_dim) if spectral_bins > 0 else None
        pooled_dim = pooled_width * sum(channels)
        extra_dim = (
            (self.position.out_features if self.position is not None else 0)
            + (self.stats.out_features if self.stats is not None else 0)
            + (self.spectral.out_dim if self.spectral is not None else 0)
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim + extra_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim + extra_dim, embed_dim),
        )
        self.embed_dim = int(embed_dim)

    def embed_curve(self, curve: torch.Tensor) -> torch.Tensor:
        """Embed a denormalized [B, L] curve into the [B, embed_dim] descriptor."""
        feature_maps = self.encoder(self.encoding(curve))
        pooled = []
        for feature_map, norm in zip(feature_maps, self.stage_norms, strict=True):
            segments = [F.adaptive_avg_pool1d(feature_map, count).flatten(1) for count in self.segments]
            pooled.append(norm(torch.cat(segments, dim=1)))
        embedding = torch.cat(pooled, dim=1)
        if self.position is not None:
            embedding = torch.cat([embedding, self.position(curve)], dim=1)
        if self.stats is not None:
            embedding = torch.cat([embedding, self.stats(curve)], dim=1)
        if self.spectral is not None:
            embedding = torch.cat([embedding, self.spectral(curve)], dim=1)
        return self.projection(embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_curve(self.curve_extractor(x))

    def diagnostic_component_paths(self) -> dict[str, str]:
        return {
            "backbone_curve_extractor": "model.backbone.curve_extractor",
            "backbone_multiscale_encoding": "model.backbone.encoding",
            "backbone_temporal_stem": "model.backbone.encoder.stem",
            "backbone_temporal_stage_last": "model.backbone.encoder.stages.3",
            "backbone_position_descriptor": "model.backbone.position",
            "backbone_projection": "model.backbone.projection",
        }


class RingContextEncoder(nn.Module):
    """Summarize how neighbor responses differ from the center, per distance ring.

    Neighbor embeddings are projected to ``context_dim`` after subtracting the
    center embedding. For each distance ring the encoder emits the masked mean
    and std of those deltas plus a count fraction; coverage and normalized
    nearest-neighbor distance are appended. The ring partition makes the
    summary invariant to neighbor ordering and roughly invariant to grid
    rotation, without discarding scale information entirely.
    """

    def __init__(self, embed_dim: int, context_dim: int = 64, ring_edges: tuple[float, ...] = (2.0, 4.0, 8.0)) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError("context_dim must be positive.")
        edges = tuple(float(edge) for edge in ring_edges)
        if not edges or any(edge <= 0 for edge in edges) or any(
            earlier >= later for earlier, later in zip(edges, edges[1:])
        ):
            raise ValueError(f"ring_edges must be positive and strictly increasing, got {edges}")
        self.ring_edges = edges
        self.delta_proj = nn.Linear(embed_dim, context_dim)
        self.out_dim = len(edges) * (2 * context_dim + 1) + 2

    def forward(
        self,
        center_embed: torch.Tensor,
        neighbor_embed: torch.Tensor,
        distances: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta_proj(neighbor_embed - center_embed.unsqueeze(1))
        neighbor_count = mask.shape[1]
        features = []
        lower = 0.0
        for edge in self.ring_edges:
            ring = mask & (distances > lower) & (distances <= edge)
            lower = edge
            weight = ring.float() / ring.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (delta * weight.unsqueeze(-1)).sum(dim=1)
            variance = (((delta - mean.unsqueeze(1)) ** 2) * weight.unsqueeze(-1)).sum(dim=1)
            features.extend(
                (
                    mean,
                    variance.clamp_min(1e-8).sqrt(),
                    ring.sum(dim=1, keepdim=True).float() / neighbor_count,
                )
            )
        coverage = mask.float().mean(dim=1, keepdim=True)
        min_distance = distances.masked_fill(~mask, float("inf")).amin(dim=1, keepdim=True)
        min_distance = torch.nan_to_num(min_distance / self.ring_edges[-1], nan=1.0, posinf=1.0, neginf=1.0)
        return torch.cat([*features, coverage, min_distance], dim=1), coverage


class ContextGateFusion(nn.Module):
    """Zero-initialized residual correction scaled by neighbor coverage.

    The fusion layer starts at exactly zero, so the module begins as the
    single-point backbone and learns how much context to use; ``coverage``
    zeroes the correction entirely when a sample has no valid neighbors.
    """

    def __init__(self, embed_dim: int, context_dim: int) -> None:
        super().__init__()
        self.fuse = nn.Linear(embed_dim + context_dim, embed_dim)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, center_embed: torch.Tensor, context: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        return center_embed + coverage * self.fuse(torch.cat([center_embed, context], dim=-1))


class TGPThermoGeoBackbone(nn.Module):
    """TGP thermo-sequence backbone conditioned on neighbor response differences.

    The center image and every neighbor curve share the same
    :class:`TGPThermoSeqBackbone` weights. The ring-encoded context applies a
    gated residual correction to the center embedding. Plain tensor inputs
    (prediction paths without coordinates) fall back to the single-point
    descriptor unchanged.
    """

    expects_dual_resolution = False
    expects_neighborhood = True

    def __init__(
        self,
        *,
        context_enabled: bool = True,
        context_dim: int = 64,
        ring_edges: tuple[float, ...] = (2.0, 4.0, 8.0),
        **sequence_kwargs,
    ) -> None:
        super().__init__()
        self.sequence = TGPThermoSeqBackbone(**sequence_kwargs)
        self.context_enabled = bool(context_enabled)
        if self.context_enabled:
            self.context = RingContextEncoder(self.sequence.embed_dim, context_dim, ring_edges)
            self.fusion = ContextGateFusion(self.sequence.embed_dim, self.context.out_dim)
        self.embed_dim = self.sequence.embed_dim

    def forward(self, x: torch.Tensor | NeighborhoodInput) -> torch.Tensor:
        if isinstance(x, NeighborhoodInput):
            center, curves, distances, mask = x
        else:
            center, curves, distances, mask = x, None, None, None
        center_embed = self.sequence(center)
        if not self.context_enabled or curves is None or curves.shape[1] == 0:
            return center_embed
        batch, count, length = curves.shape
        neighbor_curves = (curves.clamp(-1.0, 1.0) + 1.0) * 0.5
        neighbor_embed = self.sequence.embed_curve(neighbor_curves.reshape(batch * count, length))
        neighbor_embed = neighbor_embed.reshape(batch, count, self.embed_dim)
        context, coverage = self.context(center_embed, neighbor_embed, distances.float(), mask.to(torch.bool))
        return self.fusion(center_embed, context, coverage)

    def diagnostic_component_paths(self) -> dict[str, str]:
        paths = {
            f"backbone_sequence_{key}": value.replace("model.backbone", "model.backbone.sequence", 1)
            for key, value in self.sequence.diagnostic_component_paths().items()
        }
        if self.context_enabled:
            paths["backbone_ring_context"] = "model.backbone.context"
            paths["backbone_context_fusion"] = "model.backbone.fusion"
        return paths

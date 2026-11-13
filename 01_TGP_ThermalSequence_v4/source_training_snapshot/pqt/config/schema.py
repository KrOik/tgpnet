from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SplitConfig:
    strategy: str = "spatial_block"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    fold: int = 0
    num_folds: int = 4
    guard_radius: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in {"spatial_block", "region_block", "level_holdout", "full", "random"}:
            raise ValueError(f"Unsupported split strategy: {self.strategy}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
        if self.strategy == "level_holdout":
            if self.num_folds < 2:
                raise ValueError("level_holdout requires num_folds >= 2 (one fold per depth level).")
        elif self.num_folds != 4:
            raise ValueError("The canonical spatial protocol requires exactly four folds.")
        if not 0 <= self.fold < self.num_folds:
            raise ValueError(f"fold must be in [0, {self.num_folds}), got {self.fold}.")
        if self.guard_radius < 0:
            raise ValueError("guard_radius must be non-negative.")


@dataclass(slots=True)
class DataConfig:
    csv_path: Path | None = None
    image_dir: Path | None = None
    filename_column: str = "Image_Path"
    label_column: str = "Label"
    image_size: tuple[int, int] = (1024, 1024)
    sr_image_dir: Path | None = None
    sr_manifest_path: Path | None = None
    input_mode: str = "grayscale"
    train_contrast_factor: float | None = None
    train_curve_augment: float = 0.0
    train_curve_time_warp: float = 0.0
    val_curve_time_warp: float = 0.0
    context_neighbor_count: int = 0
    context_radius: float = 8.0
    export_splits: bool = False
    split_output_dir: Path | None = None
    valid_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    shuffle_train_labels: bool = False

    def __post_init__(self) -> None:
        if self.csv_path is not None:
            self.csv_path = Path(self.csv_path)
        if self.image_dir is not None:
            self.image_dir = Path(self.image_dir)
        if self.sr_image_dir is not None:
            self.sr_image_dir = Path(self.sr_image_dir)
        if self.sr_manifest_path is not None:
            self.sr_manifest_path = Path(self.sr_manifest_path)
        if self.split_output_dir is not None:
            self.split_output_dir = Path(self.split_output_dir)
        if self.train_contrast_factor is not None and self.train_contrast_factor <= 0:
            raise ValueError("train_contrast_factor must be positive when provided.")
        if not 0.0 <= self.train_curve_augment <= 1.0:
            raise ValueError("train_curve_augment must be within [0, 1].")
        if not 0.0 <= self.train_curve_time_warp <= 0.7:
            raise ValueError("train_curve_time_warp must be within [0, 0.7].")
        if not 0.0 <= self.val_curve_time_warp <= 0.7:
            raise ValueError("val_curve_time_warp must be within [0, 0.7].")
        if self.context_neighbor_count < 0:
            raise ValueError("context_neighbor_count must be non-negative.")
        if self.context_neighbor_count > 0 and self.context_radius <= 0:
            raise ValueError("context_radius must be positive when context neighbors are enabled.")


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str = "tgp_pqt_convnext_v1"
    head_name: str = "pqt_rrh"
    head_variant: str = "pqt_rrh"
    neck_name: str = "pyramid_descriptor"
    calibration_name: str = "identity"
    input_channels: int = 1
    num_outputs: int = 1
    pretrained: bool = True
    pretrained_backbone_path: Path | None = None
    sr_model_name: str = "edsr_baseline"
    sr_scale: int = 2
    sr_weights_path: Path | None = None
    sr_frozen: bool = True
    hidden_dim: int = 512
    se_reduction: int = 16
    multiscale_projection_dim: int = 128
    sra_encoding: bool = True
    cross_scale_refinement: bool = True
    se_refinement: bool = True
    thermal_scales: tuple[float, ...] = (4.0, 16.0, 64.0)
    thermal_stage_channels: tuple[int, ...] = (32, 64, 128, 192)
    thermal_pool_segments: tuple[int, ...] = (1, 2, 4)
    thermal_embed_dim: int = 256
    thermal_multiscale_encoding: bool = True
    thermal_include_raw: bool = True
    thermal_position_features: bool = True
    thermal_context_enabled: bool = True
    thermal_context_dim: int = 64
    thermal_context_ring_edges: tuple[float, ...] = (2.0, 4.0, 8.0)
    thermal_soft_position: bool = False
    thermal_global_stats: bool = False
    thermal_spectral_bins: int = 0
    thermal_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.pretrained_backbone_path is not None:
            self.pretrained_backbone_path = Path(self.pretrained_backbone_path)
        if self.sr_weights_path is not None:
            self.sr_weights_path = Path(self.sr_weights_path)
        if self.sr_model_name != "edsr_baseline":
            raise ValueError(f"Unsupported SR model: {self.sr_model_name}")
        if self.sr_scale != 2:
            raise ValueError("The canonical dual-resolution model requires SR scale 2.")
        if not self.sr_frozen:
            raise ValueError("The canonical EDSR component must remain frozen.")
        if self.head_variant not in {"pqt_rrh", "pqt_linear"}:
            raise ValueError(f"Unsupported head variant: {self.head_variant}")
        if self.multiscale_projection_dim < 1:
            raise ValueError("multiscale_projection_dim must be positive.")
        if not self.thermal_scales or any(scale <= 0 for scale in self.thermal_scales):
            raise ValueError("thermal_scales must be non-empty positive values.")
        if len(self.thermal_stage_channels) < 2 or any(width < 1 for width in self.thermal_stage_channels):
            raise ValueError("thermal_stage_channels requires at least two positive widths.")
        if not self.thermal_pool_segments or any(segment < 1 for segment in self.thermal_pool_segments):
            raise ValueError("thermal_pool_segments must be non-empty positive integers.")
        if self.thermal_embed_dim < 1:
            raise ValueError("thermal_embed_dim must be positive.")
        if self.thermal_context_dim < 1:
            raise ValueError("thermal_context_dim must be positive.")
        if self.thermal_spectral_bins < 0:
            raise ValueError("thermal_spectral_bins must be non-negative.")
        if not 0.0 <= self.thermal_dropout < 1.0:
            raise ValueError("thermal_dropout must be in [0, 1).")
        if not self.thermal_context_ring_edges or any(
            earlier >= later
            for earlier, later in zip(self.thermal_context_ring_edges, self.thermal_context_ring_edges[1:])
        ) or self.thermal_context_ring_edges[0] <= 0:
            raise ValueError("thermal_context_ring_edges must be positive and strictly increasing.")


@dataclass(slots=True)
class OptimizationConfig:
    epochs: int = 100
    validation_interval: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    clip_grad: bool = False
    max_grad_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    warmup_epochs: int = 5
    backbone_lr_multiplier: float = 0.1
    neck_lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.0
    save_interval: int = 5
    resume_state_path: Path | None = None
    segment_end_epoch: int | None = None
    loss_name: str = "huber"
    huber_delta: float = 0.08
    quantile_tau: float = 0.5
    depth_weight_gamma: float = 0.0
    depth_max: float | None = None
    ordinal_loss_weight: float = 0.001
    ema_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.loss_name not in {"mse", "huber"}:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if not 0 < self.quantile_tau < 1:
            raise ValueError("quantile_tau must be in (0, 1).")
        if self.depth_weight_gamma < 0:
            raise ValueError("depth_weight_gamma must be non-negative.")
        if self.depth_max is not None and self.depth_max <= 0:
            raise ValueError("depth_max must be positive when provided.")
        if self.ordinal_loss_weight < 0:
            raise ValueError("ordinal_loss_weight must be non-negative.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if self.backbone_lr_multiplier <= 0 or self.neck_lr_multiplier <= 0 or self.head_lr_multiplier <= 0:
            raise ValueError("learning-rate multipliers must be positive.")
        if self.resume_state_path is not None:
            self.resume_state_path = Path(self.resume_state_path)
        if self.segment_end_epoch is not None and not 1 <= self.segment_end_epoch <= self.epochs:
            raise ValueError("segment_end_epoch must be in [1, epochs] when provided.")


@dataclass(slots=True)
class RuntimeConfig:
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = False
    device: str = "auto"
    gpu_id: int = 0
    deterministic: bool = True
    optimization_seed: int | None = None
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.optimization_seed is not None and self.optimization_seed < 0:
            raise ValueError("optimization_seed must be non-negative when provided.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")

    def effective_optimization_seed(self, split_seed: int) -> int:
        return split_seed if self.optimization_seed is None else self.optimization_seed


@dataclass(slots=True)
class DiagnosticsConfig:
    """Three-speed observability configuration shared by training and diagnosis."""

    enabled: bool = False
    telemetry_enabled: bool = True
    snapshot_interval: int = 5
    snapshot_first_epoch: bool = True
    batches_per_snapshot: int = 1
    samples_per_snapshot: int = 1
    trigger_on_anomaly: bool = True
    checkpoint_path: Path | None = None
    run_ablations: bool = False
    output_subdir: str = "observability"
    collapse_patience: int = 2
    near_zero_threshold: float = 1e-6
    dominance_threshold: float = 0.90
    effective_count_threshold: float = 1.5

    @property
    def epoch_interval(self) -> int:
        """Compatibility alias for configurations written before v0.2."""
        return self.snapshot_interval

    @epoch_interval.setter
    def epoch_interval(self, value: int) -> None:
        self.snapshot_interval = value

    def __post_init__(self) -> None:
        if self.snapshot_interval < 1:
            raise ValueError("diagnostics.snapshot_interval must be at least 1.")
        if self.batches_per_snapshot < 1:
            raise ValueError("diagnostics.batches_per_snapshot must be at least 1.")
        if self.samples_per_snapshot < 1:
            raise ValueError("diagnostics.samples_per_snapshot must be at least 1.")
        if self.collapse_patience < 1:
            raise ValueError("diagnostics.collapse_patience must be at least 1.")
        if not 0 < self.dominance_threshold <= 1:
            raise ValueError("diagnostics.dominance_threshold must be in (0, 1].")
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        if not self.output_subdir or Path(self.output_subdir).is_absolute() or ".." in Path(self.output_subdir).parts:
            raise ValueError("diagnostics.output_subdir must be a safe relative path.")


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path = Path("outputs")
    best_checkpoint_name: str = "efficientnet_v2_best.pth"
    final_checkpoint_name: str = "efficientnet_v2_final.pth"
    training_history_name: str = "training_history.csv"
    training_plot_name: str = "training_plot.png"
    evaluation_results_name: str = "evaluation_results.csv"
    evaluation_plot_name: str = "evaluation_plots.png"
    prediction_plot_name: str = "prediction_scatter.png"
    prediction_csv_name: str = "predictions.csv"
    metrics_summary_name: str = "metrics_summary.txt"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str | None = None
    root_dir: Path = Path("experiments")
    validate_branch: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)


@dataclass(slots=True)
class BestCheckpointConfig:
    strategy: str = "val_mae"

    def __post_init__(self) -> None:
        valid_strategies = {"val_mae", "val_loss", "train_loss", "fixed_epochs"}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Unsupported best checkpoint strategy: {self.strategy}")


@dataclass(slots=True)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    best_checkpoint: BestCheckpointConfig = field(default_factory=BestCheckpointConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass(slots=True)
class EvaluationConfig:
    checkpoint_path: Path | None = None
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    split_name: str = "test"
    external_datasets: tuple["StandardEvaluationDatasetSpec", ...] = field(default_factory=tuple)
    include_same_surface: bool = False
    include_pointwise_diagnostics: bool = False
    include_geometry: bool = True
    include_partitions: bool = True
    render_enabled: bool = False
    render_timeout_seconds: int = 300
    matlab_executable: str | None = None
    bootstrap_tile_size: int = 8
    bootstrap_repetitions: int = 2000
    tolerance: float = 0.08

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        self.external_datasets = tuple(self.external_datasets)
        if self.bootstrap_tile_size < 1:
            raise ValueError("bootstrap_tile_size must be positive.")
        if self.bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(slots=True)
class PredictionConfig:
    checkpoint_path: Path | None = None
    image_dir: Path | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    input_mode: str = "grayscale"

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.image_dir is not None:
            self.image_dir = Path(self.image_dir)


@dataclass(slots=True)
class StandardEvaluationDatasetSpec:
    name: str
    image_dir: Path
    ground_truth_csv: Path | None = None
    input_mode: str = "grayscale"
    sr_image_dir: Path | None = None
    sr_manifest_path: Path | None = None
    valid_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    def __post_init__(self) -> None:
        self.image_dir = Path(self.image_dir)
        if self.sr_image_dir is not None:
            self.sr_image_dir = Path(self.sr_image_dir)
        if self.sr_manifest_path is not None:
            self.sr_manifest_path = Path(self.sr_manifest_path)
        if self.ground_truth_csv is not None:
            self.ground_truth_csv = Path(self.ground_truth_csv)

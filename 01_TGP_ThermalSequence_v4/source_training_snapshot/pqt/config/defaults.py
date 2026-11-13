from __future__ import annotations

from pathlib import Path

from pqt.data.sr_cache import SR_WEIGHT_RELATIVE_PATH

from .schema import (
    BestCheckpointConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    PredictionConfig,
    TrainConfig,
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_pretrained_path(asset_root: Path | None = None) -> Path | None:
    candidate = (asset_root or get_repo_root()) / "legacy" / "pqt_model" / "pretrained" / "efficientnet_v2_s.pth"
    return candidate if candidate.exists() else None


def _convnext_pretrained_path(asset_root: Path | None = None) -> Path:
    candidate = (asset_root or get_repo_root()) / "artifacts" / "pretrained" / "convnext_tiny_imagenet1k_v1.pth"
    return candidate


def build_default_train_config(
    dataset_name: str = "data1",
    asset_root: Path | None = None,
    output_dir: Path | None = None,
) -> TrainConfig:
    repo_root = get_repo_root()
    asset_root = Path(asset_root).resolve() if asset_root is not None else repo_root
    dataset_dir = asset_root / "data" / "train" / dataset_name
    return TrainConfig(
        data=DataConfig(
            csv_path=dataset_dir / f"{dataset_name}.csv",
            image_dir=dataset_dir / "images",
        ),
        model=ModelConfig(
            pretrained_backbone_path=_convnext_pretrained_path(asset_root),
            sr_weights_path=asset_root / SR_WEIGHT_RELATIVE_PATH,
        ),
        output=OutputConfig(output_dir=output_dir or (repo_root / "outputs" / dataset_name / "train")),
        best_checkpoint=BestCheckpointConfig(strategy="val_mae"),
    )


def build_default_evaluation_config(
    dataset_name: str = "data1",
    checkpoint_path: Path | None = None,
    asset_root: Path | None = None,
    output_dir: Path | None = None,
) -> EvaluationConfig:
    from pqt.registry.external_datasets import build_default_standard_evaluation_specs

    train_config = build_default_train_config(dataset_name=dataset_name, asset_root=asset_root)
    checkpoint = checkpoint_path or (train_config.output.output_dir / train_config.output.best_checkpoint_name)
    repo_root = get_repo_root()
    return EvaluationConfig(
        checkpoint_path=checkpoint,
        data=train_config.data,
        model=train_config.model,
        runtime=train_config.runtime,
        output=OutputConfig(output_dir=output_dir or (repo_root / "outputs" / dataset_name / "evaluate")),
        external_datasets=tuple(
            spec
            for spec in build_default_standard_evaluation_specs(Path(asset_root).resolve() if asset_root else repo_root)
            if spec.name.startswith("sample_")
        ),
        include_same_surface=False,
        include_pointwise_diagnostics=False,
        include_geometry=False,
        include_partitions=True,
        render_enabled=False,
    )


def build_default_prediction_config(
    image_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    asset_root: Path | None = None,
    output_dir: Path | None = None,
) -> PredictionConfig:
    repo_root = get_repo_root()
    asset_root = Path(asset_root).resolve() if asset_root is not None else repo_root
    default_image_dir = image_dir or (asset_root / "data" / "eval" / "sample_round" / "images")
    default_checkpoint = checkpoint_path
    model_config = ModelConfig(
        pretrained_backbone_path=_convnext_pretrained_path(asset_root),
        sr_weights_path=asset_root / SR_WEIGHT_RELATIVE_PATH,
        calibration_name="identity",
    )
    return PredictionConfig(
        checkpoint_path=default_checkpoint,
        image_dir=default_image_dir,
        model=model_config,
        output=OutputConfig(output_dir=output_dir or (repo_root / "outputs" / "predict")),
    )


def build_default_standard_evaluation_config(
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    dataset_name: str = "data1",
    asset_root: Path | None = None,
) -> EvaluationConfig:
    config = build_default_evaluation_config(
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        asset_root=asset_root,
        output_dir=output_dir,
    )
    return config

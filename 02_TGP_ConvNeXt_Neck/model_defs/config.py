from __future__ import annotations

from pqt.config.schema import BestCheckpointConfig, TrainConfig

EXPERIMENT_ID = "tgp_convnext_neck"


def _bind_experiment_outputs(config: TrainConfig) -> None:
    if config.output.output_dir.name != EXPERIMENT_ID:
        config.output.output_dir = config.output.output_dir / EXPERIMENT_ID
    config.output.best_checkpoint_name = f"{EXPERIMENT_ID}_v0.4.0_best.pth"
    config.output.final_checkpoint_name = f"{EXPERIMENT_ID}_v0.4.0_final.pth"


def apply_train_config(config: TrainConfig) -> TrainConfig:
    config.experiment.experiment_id = EXPERIMENT_ID
    selection_strategy = "train_loss" if config.split.strategy == "full" else "val_mae"
    config.best_checkpoint = BestCheckpointConfig(strategy=selection_strategy)
    config.runtime.num_workers = min(config.runtime.num_workers, 4)
    config.data.image_size = (512, 512)
    config.model.hidden_dim = max(config.model.hidden_dim, 896)
    config.model.pretrained = False
    config.model.pretrained_backbone_path = None
    if config.optimization.epochs == 100:
        config.optimization.epochs = 40
    config.optimization.validation_interval = 1
    config.optimization.backbone_lr_multiplier = 0.2
    _bind_experiment_outputs(config)
    return config

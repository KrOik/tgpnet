from __future__ import annotations

from pqt.config.schema import BestCheckpointConfig, TrainConfig


EXPERIMENT_ID = "tgp-thermo-seq-v4-noraw"


def apply_train_config(config: TrainConfig) -> TrainConfig:
    config.experiment.experiment_id = EXPERIMENT_ID
    config.model.backbone_name = "tgp_thermo_seq_v2"
    config.model.head_name = "pqt_rrh"
    config.model.head_variant = "pqt_rrh"
    config.model.neck_name = "identity"
    config.model.pretrained = False
    config.model.thermal_scales = (4.0, 16.0, 64.0)
    config.model.thermal_stage_channels = (32, 64, 128, 192)
    config.model.thermal_pool_segments = (1, 2, 4)
    config.model.thermal_embed_dim = 256
    config.model.thermal_multiscale_encoding = True
    config.model.thermal_include_raw = False
    config.model.thermal_position_features = True
    config.model.thermal_soft_position = False
    config.model.thermal_global_stats = True
    config.model.thermal_spectral_bins = 32
    config.model.thermal_dropout = 0.1
    config.data.input_mode = "curve"
    config.split.guard_radius = 0
    config.optimization.loss_name = "huber"
    config.optimization.huber_delta = 0.08
    config.optimization.mse_weight = 0.0
    config.optimization.l1_weight = 0.0
    config.optimization.ordinal_loss_weight = 0.05
    config.optimization.ema_decay = 0.999
    if config.optimization.epochs == 100:
        config.optimization.epochs = 40
    config.optimization.validation_interval = 1
    config.optimization.backbone_lr_multiplier = 1.0
    config.best_checkpoint = BestCheckpointConfig(
        strategy="train_loss" if config.split.strategy == "full" else "val_mae"
    )
    config.output.best_checkpoint_name = f"{EXPERIMENT_ID}_v1.0.0_best.pth"
    config.output.final_checkpoint_name = f"{EXPERIMENT_ID}_v1.0.0_final.pth"
    return config

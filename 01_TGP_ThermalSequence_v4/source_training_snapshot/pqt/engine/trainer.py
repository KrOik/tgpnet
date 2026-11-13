from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pqt.config.schema import TrainConfig
from pqt.data.loaders import create_supervised_dataloaders
from pqt.diagnostics import finalize_training_snapshot, trace_training_batch
from pqt.engine.losses import compute_loss_breakdown
from pqt.experiments.loader import apply_experiment_train_config, build_training_loss, resolve_experiment
from pqt.experiments.contracts import ExperimentRuntime
from pqt.model.checkpoint import load_trusted_checkpoint_payload, save_checkpoint
from pqt.model.factory import build_model
from pqt.model.inputs import model_input_batch_size, move_model_input, slice_model_input
from pqt.model.outputs import RegressionOutput
from pqt.observation import observation_sink
from pqt.observability import ObserverHub
from pqt.viz.training import plot_training_history

from .metrics import (
    calculate_offgrid_metrics,
    calculate_ordered_depth_metrics,
    calculate_regression_metrics,
)
from .runtime import resolve_device, seed_runtime
from .schedulers import WarmupScheduler


DEPTH_SLICES = {
    "shallow": (0.24, 0.40, 0.56),
    "middle": (0.72, 0.88, 1.04),
    "deep": (1.20, 1.36, 1.52),
}


def _depth_slice_mae(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    result: dict[str, float] = {}
    for name, depths in DEPTH_SLICES.items():
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for depth in depths:
            mask |= torch.isclose(targets, targets.new_tensor(depth), atol=1e-5, rtol=0.0)
        result[name] = float(torch.abs(predictions[mask] - targets[mask]).mean()) if mask.any() else float("nan")
    return result


def _require_canonical_training_paths(config: TrainConfig) -> None:
    if config.data.csv_path is None or config.data.image_dir is None:
        raise ValueError("Canonical training requires data/train/data1 CSV and image paths.")
    csv_path = Path(config.data.csv_path).resolve()
    image_dir = Path(config.data.image_dir).resolve()
    if csv_path.parts[-4:] != ("data", "train", "data1", "data1.csv"):
        raise ValueError(f"Training CSV must be canonical data/train/data1/data1.csv: {csv_path}")
    if image_dir.parts[-4:] != ("data", "train", "data1", "images"):
        raise ValueError(f"Training images must be canonical data/train/data1/images: {image_dir}")
    if csv_path.parents[3] != image_dir.parents[3]:
        raise ValueError("Training CSV and images must belong to the same canonical asset root.")


def _build_optimizer(model: nn.Module, criterion: nn.Module, config: TrainConfig) -> optim.Optimizer:
    if hasattr(model, "parameter_groups"):
        component_groups = model.parameter_groups()
    else:
        component_groups = {
            "model": [parameter for parameter in model.parameters() if parameter.requires_grad],
        }

    learning_rates = {
        "input_fusion": config.optimization.lr,
        "backbone": config.optimization.lr * config.optimization.backbone_lr_multiplier,
        "neck": config.optimization.lr * config.optimization.neck_lr_multiplier,
        "head": config.optimization.lr * config.optimization.head_lr_multiplier,
        "calibrator": config.optimization.lr,
        "model": config.optimization.lr,
        "criterion": config.optimization.lr,
    }
    component_groups["criterion"] = [parameter for parameter in criterion.parameters() if parameter.requires_grad]
    ordered_groups = ["input_fusion", "backbone", "neck", "head", "calibrator", "model", "criterion"]
    optimizer_groups = []
    for group_name in ordered_groups:
        params = [parameter for parameter in component_groups.get(group_name, []) if parameter.requires_grad]
        if not params:
            continue
        target_lr = learning_rates[group_name]
        optimizer_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": target_lr,
                "target_lr": target_lr,
                "warmup_initial_lr": target_lr * config.optimization.backbone_lr_multiplier,
            }
        )

    return optim.AdamW(optimizer_groups, weight_decay=config.optimization.weight_decay)


def _resume_contract(config: TrainConfig) -> dict[str, object]:
    payload = asdict(config)
    optimization = dict(payload["optimization"])
    optimization.pop("resume_state_path", None)
    optimization.pop("segment_end_epoch", None)
    payload["optimization"] = optimization
    return payload


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_training_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_state(
    path: Path,
    config: TrainConfig,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    warmup: WarmupScheduler,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    payload = load_trusted_checkpoint_payload(path)
    required = {
        "resume_schema_version", "epoch", "contract", "model_state_dict", "optimizer_state_dict",
        "criterion_state_dict", "scheduler_state_dict", "warmup_state_dict", "scaler_state_dict",
        "history", "best_metric_value", "best_metric_name", "best_epoch", "rng_state",
    }
    if not isinstance(payload, dict) or payload.get("resume_schema_version") != 1 or not required <= set(payload):
        raise ValueError(f"Invalid training resume state: {path}")
    if payload["contract"] != _resume_contract(config):
        raise ValueError("Resume state contract does not match the requested training configuration.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    criterion.load_state_dict(payload["criterion_state_dict"], strict=True)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    warmup.load_state_dict(payload["warmup_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return payload


def train_model(
    config: TrainConfig,
    experiment: ExperimentRuntime | None = None,
    *,
    apply_experiment_config: bool = True,
) -> dict[str, object]:
    experiment_runtime = experiment or resolve_experiment(config.experiment)
    active_config = (
        apply_experiment_train_config(config, experiment_runtime)
        if apply_experiment_config
        else deepcopy(config)
    )
    optimization_seed = active_config.runtime.optimization_seed
    effective_optimization_seed = active_config.runtime.effective_optimization_seed(active_config.split.seed)
    seed_runtime(effective_optimization_seed, active_config.runtime.deterministic)
    _require_canonical_training_paths(active_config)
    device = resolve_device(active_config.runtime)
    active_config.output.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        active_config.model,
        assembly_overrides=experiment_runtime.assembly_overrides if experiment_runtime else None,
    ).to(device)
    loaders = create_supervised_dataloaders(active_config.data, active_config.split, active_config.runtime)
    train_frame = getattr(getattr(loaders.train, "dataset", None), "dataframe", None)
    if train_frame is not None:
        active_config.optimization.depth_max = float(train_frame[active_config.data.label_column].astype(float).max())
    split_audit = {
        "strategy": active_config.split.strategy,
        "fold": active_config.split.fold,
        "num_folds": active_config.split.num_folds,
        "guard_radius": active_config.split.guard_radius,
        "seed": active_config.split.seed,
        "split_seed": active_config.split.seed,
        "optimization_seed": optimization_seed,
        "effective_optimization_seed": effective_optimization_seed,
        "shuffle_train_labels_negative_control": active_config.data.shuffle_train_labels,
        "train_sample_size": len(getattr(loaders.train, "dataset", loaders.train)),
        "validation_sample_size": len(getattr(loaders.val, "dataset", loaders.val)),
        "test_sample_size": len(getattr(loaders.test, "dataset", loaders.test)),
        "external_data_access": False,
        "generalization_claim": "internal optimization only; external geometry evaluation required",
        "depth_max": active_config.optimization.depth_max,
    }
    (active_config.output.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    criterion = build_training_loss(active_config, experiment_runtime).to(device)
    optimizer = _build_optimizer(model, criterion, active_config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=active_config.optimization.scheduler_factor,
        patience=active_config.optimization.scheduler_patience,
    )
    warmup = WarmupScheduler(
        optimizer,
        warmup_epochs=active_config.optimization.warmup_epochs,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "learning_rates": [],
        "prediction_bias": [],
        "val_rmse": [],
        "val_bias": [],
        "val_depth_class_accuracy": [],
        "val_depth_within_one_class": [],
        "val_depth_mean_class_distance": [],
        "val_mae_shallow": [],
        "val_mae_middle": [],
        "val_mae_deep": [],
        "val_off_grid_rate": [],
        "val_within_level_std": [],
    }
    optimizer_group_names = [str(group.get("name", f"group_{index}")) for index, group in enumerate(optimizer.param_groups)]
    for group_name in optimizer_group_names:
        history[f"lr_{group_name}"] = []
    best_path = active_config.output.output_dir / active_config.output.best_checkpoint_name
    final_path = active_config.output.output_dir / active_config.output.final_checkpoint_name
    best_metric_value = float("inf")
    best_metric_name = active_config.best_checkpoint.strategy
    best_epoch = 0
    training_start_time = perf_counter()
    observer = ObserverHub(
        active_config.output.output_dir,
        active_config.diagnostics,
        model,
        criterion,
        optimizer,
        run_id=active_config.output.output_dir.name,
    ) if active_config.diagnostics.telemetry_enabled else None
    if observer is not None:
        observer.train_start()

    full_training = active_config.split.strategy == "full"
    if full_training and best_metric_name != "train_loss":
        raise ValueError("full training must select the internal best train_loss checkpoint.")
    if not full_training and best_metric_name not in {"val_mae", "val_loss"}:
        raise ValueError("non-full training requires val_mae or val_loss checkpoint selection.")
    validation_interval = active_config.optimization.validation_interval
    if validation_interval < 1:
        raise ValueError("optimization.validation_interval must be at least 1.")
    accumulation_steps = active_config.runtime.gradient_accumulation_steps
    amp_enabled = active_config.runtime.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    state_path = active_config.optimization.resume_state_path or (
        active_config.output.output_dir / "training_state.pth"
    )
    start_epoch = 0
    if active_config.optimization.resume_state_path is not None:
        resume_payload = _load_resume_state(
            state_path, active_config, model, optimizer, criterion, scheduler, warmup, scaler
        )
        start_epoch = int(resume_payload["epoch"])
        history = resume_payload["history"]
        for name in (
            "val_depth_class_accuracy",
            "val_depth_within_one_class",
            "val_depth_mean_class_distance",
        ):
            history.setdefault(name, [float("nan")] * start_epoch)
        best_metric_value = float(resume_payload["best_metric_value"])
        best_metric_name = str(resume_payload["best_metric_name"])
        best_epoch = int(resume_payload["best_epoch"])
        if start_epoch >= active_config.optimization.epochs:
            raise ValueError("Resume state already reached the configured final epoch.")
    segment_end_epoch = active_config.optimization.segment_end_epoch or active_config.optimization.epochs
    if segment_end_epoch <= start_epoch:
        raise ValueError("segment_end_epoch must be greater than the resumed epoch.")

    ema_decay = float(active_config.optimization.ema_decay)
    ema_state: dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, segment_end_epoch):
        epoch_start_time = perf_counter()
        model.train()
        epoch_loss = 0.0
        train_predictions: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(loaders.train):
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loaders.train)
            inputs = move_model_input(inputs, device)
            targets = targets.to(device)
            should_trace = observer is not None and observer.should_snapshot(epoch + 1, batch_index + 1)
            traced = None
            if should_trace:
                snapshot_dir = (
                    active_config.output.output_dir
                    / active_config.diagnostics.output_subdir
                    / "snapshots"
                    / f"epoch-{epoch + 1:04d}-batch-{batch_index + 1:04d}"
                )
                sample_count = min(active_config.diagnostics.samples_per_snapshot, model_input_batch_size(inputs))
                traced = trace_training_batch(
                    model,
                    criterion,
                    slice_model_input(inputs, sample_count),
                    targets[:sample_count],
                    snapshot_dir,
                )
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                with observation_sink(observer.sink if observer is not None else None):
                    model_output = model.forward_with_aux(inputs) if hasattr(model, "forward_with_aux") else RegressionOutput(model(inputs))
                outputs = model_output.prediction
                breakdown = compute_loss_breakdown(criterion, model_output, targets)
                loss = breakdown.total
            if observer is not None:
                observer.batch_forward(model_output, targets, track_optimization=should_step)
                observer.loss_ready(breakdown)
            scaler.scale(loss / accumulation_steps).backward()
            if observer is not None and should_step:
                observer.backward_done()
            if should_step:
                if active_config.optimization.clip_grad:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [parameter for group in optimizer.param_groups for parameter in group["params"]],
                        active_config.optimization.max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                if observer is not None:
                    observer.optimizer_done()
                if ema_decay > 0.0:
                    current_state = model.state_dict()
                    if ema_state is None:
                        ema_state = {
                            name: value.detach().clone()
                            for name, value in current_state.items()
                            if torch.is_floating_point(value)
                        }
                    else:
                        for name, value in current_state.items():
                            if name in ema_state:
                                ema_state[name].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                optimizer.zero_grad(set_to_none=True)
            if traced is not None:
                finalize_training_snapshot(traced, model, criterion, epoch + 1, batch_index + 1)
            epoch_loss += loss.item()
            train_predictions.append(outputs.detach().cpu())
            train_targets.append(targets.detach().cpu())

        train_loss = epoch_loss / max(1, len(loaders.train))
        history["train_loss"].append(train_loss)

        train_metrics = calculate_regression_metrics(torch.cat(train_predictions), torch.cat(train_targets))
        history["prediction_bias"].append(train_metrics["bias"])

        group_lrs = {str(group.get("name", f"group_{index}")): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
        history["learning_rates"].append(group_lrs.get("backbone", float(optimizer.param_groups[0]["lr"])))
        for group_name in optimizer_group_names:
            history[f"lr_{group_name}"].append(group_lrs[group_name])
        epoch_number = epoch + 1
        should_validate = not full_training and (
            epoch_number % validation_interval == 0 or epoch_number == active_config.optimization.epochs
        )
        epoch_log_parts = [
            f"epoch={epoch_number}/{active_config.optimization.epochs}",
            f"train_loss={train_loss:.6f}",
            f"train_bias={train_metrics['bias']:.6f}",
            f"lr={group_lrs.get('backbone', float(optimizer.param_groups[0]['lr'])):.8f}",
        ]

        eval_backup: dict[str, torch.Tensor] | None = None
        if not should_validate:
            history["val_loss"].append(float("nan"))
            history["val_mae"].append(float("nan"))
            history["val_rmse"].append(float("nan"))
            history["val_bias"].append(float("nan"))
            history["val_depth_class_accuracy"].append(float("nan"))
            history["val_depth_within_one_class"].append(float("nan"))
            history["val_depth_mean_class_distance"].append(float("nan"))
            history["val_off_grid_rate"].append(float("nan"))
            history["val_within_level_std"].append(float("nan"))
            for slice_name in DEPTH_SLICES:
                history[f"val_mae_{slice_name}"].append(float("nan"))
            warmup.step()
            epoch_log_parts.append("validation=skipped")
        else:
            if ema_state is not None:
                eval_backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
                model.load_state_dict(ema_state, strict=False)
            model.eval()
            val_loss_total = 0.0
            val_predictions: list[torch.Tensor] = []
            val_targets: list[torch.Tensor] = []
            with torch.no_grad():
                for inputs, targets in loaders.val:
                    inputs = move_model_input(inputs, device)
                    targets = targets.to(device)
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        model_output = model.forward_with_aux(inputs) if hasattr(model, "forward_with_aux") else RegressionOutput(model(inputs))
                        outputs = model_output.prediction
                        loss = criterion(model_output, targets)
                    val_loss_total += loss.item()
                    val_predictions.append(outputs.cpu())
                    val_targets.append(targets.cpu())

            val_loss = val_loss_total / max(1, len(loaders.val))
            history["val_loss"].append(val_loss)
            all_val_predictions = torch.cat(val_predictions)
            all_val_targets = torch.cat(val_targets)
            val_metrics = calculate_regression_metrics(all_val_predictions, all_val_targets)
            depth_metrics = calculate_ordered_depth_metrics(all_val_predictions, all_val_targets)
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_bias"].append(val_metrics["bias"])
            history["val_depth_class_accuracy"].append(depth_metrics["depth_class_accuracy"])
            history["val_depth_within_one_class"].append(depth_metrics["depth_within_one_class"])
            history["val_depth_mean_class_distance"].append(depth_metrics["depth_mean_class_distance"])
            offgrid_metrics = calculate_offgrid_metrics(all_val_predictions, all_val_targets)
            history["val_off_grid_rate"].append(offgrid_metrics["off_grid_rate"])
            history["val_within_level_std"].append(offgrid_metrics["within_level_std"])
            for slice_name, value in _depth_slice_mae(all_val_predictions, all_val_targets).items():
                history[f"val_mae_{slice_name}"].append(value)
            selection_metric_value = val_metrics["mae"] if best_metric_name == "val_mae" else val_loss
            epoch_log_parts.extend(
                [
                    f"val_loss={val_loss:.6f}",
                    f"val_bias={val_metrics['bias']:.6f}",
                    f"val_rmse={val_metrics['rmse']:.6f}",
                    f"val_mae={val_metrics['mae']:.6f}",
                    f"val_depth_acc={depth_metrics['depth_class_accuracy']:.4f}",
                    f"val_within1={depth_metrics['depth_within_one_class']:.4f}",
                ]
            )
            if warmup.is_warmup_complete():
                scheduler.step(val_loss)
            else:
                warmup.step()

        checkpoint_metrics: dict[str, float | str | int | None] = {
            "train_loss": train_loss,
            "split_seed": active_config.split.seed,
            "optimization_seed": optimization_seed,
            "effective_optimization_seed": effective_optimization_seed,
        }
        if should_validate:
            checkpoint_metrics.update({"val_loss": val_loss, **val_metrics})
        selection_metric_value = train_loss if full_training else selection_metric_value
        if selection_metric_value < best_metric_value:
            best_metric_value = selection_metric_value
            best_epoch = epoch_number
            epoch_log_parts.append("best_checkpoint_updated=true")
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch_number,
                metrics=checkpoint_metrics,
                config=asdict(active_config),
            )

        if eval_backup is not None:
            model.load_state_dict(eval_backup)

        if observer is not None:
            observer.epoch_end(epoch + 1)

        _save_training_state(
            state_path,
            {
                "resume_schema_version": 1,
                "epoch": epoch_number,
                "contract": _resume_contract(active_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "warmup_state_dict": warmup.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_metric_value": best_metric_value,
                "best_metric_name": best_metric_name,
                "best_epoch": best_epoch,
                "rng_state": _capture_rng_state(),
            },
        )

        epoch_elapsed_seconds = perf_counter() - epoch_start_time
        total_elapsed_seconds = perf_counter() - training_start_time
        epoch_log_parts.append(f"best_metric_name={best_metric_name}")
        epoch_log_parts.append(f"best_metric_value={best_metric_value:.6f}")
        epoch_log_parts.append(f"best_epoch={best_epoch}")
        epoch_log_parts.append(f"epoch_seconds={epoch_elapsed_seconds:.1f}")
        epoch_log_parts.append(f"total_seconds={total_elapsed_seconds:.1f}")
        print(" ".join(epoch_log_parts), flush=True)

    history_path = active_config.output.output_dir / active_config.output.training_history_name
    pd.DataFrame(history).to_csv(history_path, index=False)
    training_complete = segment_end_epoch == active_config.optimization.epochs
    if not training_complete:
        observability_artifacts = observer.train_end() if observer is not None else None
        return {
            "device": str(device), "history": history, "history_path": history_path,
            "resume_state_path": state_path, "completed_epochs": segment_end_epoch,
            "training_complete": False, "best_checkpoint_path": best_path,
            "best_checkpoint_metric": best_metric_name, "best_metric_value": best_metric_value,
            "best_epoch": best_epoch, "experiment_id": experiment_runtime.manifest.experiment_id if experiment_runtime else None,
            "experiment_version": experiment_runtime.manifest.version if experiment_runtime else None,
            "observability_profile": "snapshots" if active_config.diagnostics.enabled else ("telemetry" if observer else "off"),
            "observability_artifacts": observability_artifacts, "diagnostic_status": "completed" if observer else "disabled",
        }

    final_metrics = {
        "best_checkpoint_metric": best_metric_name,
        "best_checkpoint_epoch": float(best_epoch),
        "split_seed": active_config.split.seed,
        "optimization_seed": optimization_seed,
        "effective_optimization_seed": effective_optimization_seed,
    }
    final_metrics[f"best_{best_metric_name}"] = best_metric_value
    save_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        criterion=criterion,
        epoch=active_config.optimization.epochs,
        metrics=final_metrics,
        config=asdict(active_config),
    )
    plot_training_history(history, active_config.output.output_dir / active_config.output.training_plot_name)
    observability_artifacts = observer.train_end() if observer is not None else None
    return {
        "device": str(device),
        "history": history,
        "best_checkpoint_path": best_path,
        "final_checkpoint_path": final_path,
        "history_path": history_path,
        "external_history_path": None,
        "best_checkpoint_metric": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "experiment_id": experiment_runtime.manifest.experiment_id if experiment_runtime else None,
        "experiment_version": experiment_runtime.manifest.version if experiment_runtime else None,
        "observability_profile": "snapshots" if active_config.diagnostics.enabled else ("telemetry" if observer else "off"),
        "observability_artifacts": observability_artifacts,
        "diagnostic_status": "completed" if observer else "disabled",
        "resume_state_path": state_path,
        "completed_epochs": active_config.optimization.epochs,
        "training_complete": True,
    }

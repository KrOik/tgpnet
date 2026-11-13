from __future__ import annotations

from pqt.config.schema import TrainConfig
from pqt.data.loaders import create_supervised_dataloaders
from pqt.diagnostics import finalize_training_snapshot, run_causal_ablations, trace_training_batch
from pqt.engine.runtime import resolve_device
from pqt.experiments.contracts import ExperimentRuntime
from pqt.experiments.loader import build_training_loss
from pqt.model.checkpoint import load_checkpoint
from pqt.model.factory import build_model
from pqt.model.inputs import move_model_input


def run_model_diagnostics(config: TrainConfig, experiment: ExperimentRuntime) -> dict[str, object]:
    if config.diagnostics.checkpoint_path is None:
        raise ValueError("Standalone diagnosis requires diagnostics.checkpoint_path.")
    device = resolve_device(config.runtime)
    model = build_model(config.model, experiment=experiment).to(device)
    criterion = build_training_loss(config, experiment).to(device)
    checkpoint_metadata = load_checkpoint(
        config.diagnostics.checkpoint_path,
        model=model,
        criterion=criterion,
        map_location=device,
        strict=True,
    )
    loaders = create_supervised_dataloaders(config.data, config.split, config.runtime)
    model.eval()
    criterion.eval()
    inputs, targets = next(iter(loaders.val))
    inputs = move_model_input(inputs, device)
    targets = targets.to(device)
    output_dir = config.output.output_dir / config.diagnostics.output_subdir
    snapshot_dir = output_dir / "checkpoint-snapshot"
    model.zero_grad(set_to_none=True)
    criterion.zero_grad(set_to_none=True)
    traced = trace_training_batch(model, criterion, inputs, targets, snapshot_dir)
    finalize_training_snapshot(traced, model, criterion, epoch=0, batch=1)
    ablation_path = None
    if config.diagnostics.run_ablations:
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        ablation_path = run_causal_ablations(model, criterion, inputs, targets, output_dir)
    return {
        "device": str(device),
        "checkpoint_path": config.diagnostics.checkpoint_path,
        "criterion_state": checkpoint_metadata.get("criterion_state", "not_requested"),
        "snapshot_dir": snapshot_dir,
        "ablation_path": ablation_path,
    }

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonify(tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen four-geometry depth-error evaluation protocol.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="data1")
    parser.add_argument("--experiment-id")
    parser.add_argument("--worktree")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--render-enabled", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    worktree = Path(args.worktree).resolve() if args.worktree else Path.cwd().resolve()

    from pqt.config.defaults import build_default_standard_evaluation_config, build_default_train_config
    from pqt.engine.evaluator import evaluate_model
    from pqt.experiments.loader import apply_experiment_train_config, resolve_experiment
    from pqt.model.checkpoint import load_trusted_checkpoint_payload, model_config_from_checkpoint

    train_config = build_default_train_config(dataset_name=args.dataset_name)
    if args.experiment_id:
        train_config.experiment.experiment_id = args.experiment_id
        train_config.experiment.validate_branch = False
        train_config.experiment.root_dir = worktree / "experiments"
    experiment_runtime = resolve_experiment(train_config.experiment)
    active_train_config = apply_experiment_train_config(train_config, experiment_runtime)

    eval_config = build_default_standard_evaluation_config(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
    )
    checkpoint_payload = load_trusted_checkpoint_payload(checkpoint_path, map_location="cpu")
    checkpoint_metadata = {key: value for key, value in checkpoint_payload.items() if key != "model_state_dict"}
    eval_config.model = model_config_from_checkpoint(checkpoint_metadata, fallback=active_train_config.model)
    # Checkpoints produced from isolated worktrees retain absolute component
    # paths.  Reuse the manifest-backed assets of this checkout when those
    # historical paths are no longer mounted; architecture fields remain those
    # embedded in the checkpoint.
    for field in ("pretrained_backbone_path", "sr_weights_path"):
        checkpoint_asset = getattr(eval_config.model, field)
        current_asset = getattr(active_train_config.model, field)
        if checkpoint_asset is not None and not Path(checkpoint_asset).is_file():
            if current_asset is None or not Path(current_asset).is_file():
                raise FileNotFoundError(f"No manifest-backed replacement for {field}: {checkpoint_asset}")
            setattr(eval_config.model, field, current_asset)
    eval_config.data.image_size = active_train_config.data.image_size
    eval_config.runtime.batch_size = (
        int(args.batch_size) if args.batch_size is not None else int(active_train_config.runtime.batch_size)
    )
    eval_config.runtime.num_workers = (
        int(args.num_workers) if args.num_workers is not None else int(active_train_config.runtime.num_workers)
    )
    use_cuda = torch.cuda.is_available()
    eval_config.runtime.pin_memory = bool(use_cuda)
    eval_config.runtime.device = "cuda" if use_cuda else "cpu"
    eval_config.runtime.gpu_id = int(args.gpu_id)
    eval_config.render_enabled = bool(args.render_enabled)
    eval_config.include_same_surface = False
    eval_config.include_pointwise_diagnostics = False

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "worktree": str(worktree),
        "experiment_id": args.experiment_id,
        "dataset_name": args.dataset_name,
        "gpu_id": args.gpu_id,
        "device": eval_config.runtime.device,
        "batch_size": eval_config.runtime.batch_size,
        "num_workers": eval_config.runtime.num_workers,
        "render_enabled": eval_config.render_enabled,
        "external_datasets": [dataset.name for dataset in eval_config.external_datasets],
    }
    print("standard_evaluation_start", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    result = evaluate_model(eval_config, experiment=experiment_runtime)

    payload = {
        **summary,
        "result": {
            "per_geometry_metrics_path": _jsonify(result.get("per_geometry_metrics_path")),
            "per_geometry_slices_path": _jsonify(result.get("per_geometry_slices_path")),
            "baseline_comparison_path": _jsonify(result.get("baseline_comparison_path")),
            "bootstrap_intervals_path": _jsonify(result.get("bootstrap_intervals_path")),
            "protocol_path": _jsonify(result.get("protocol_path")),
            "render_status_path": _jsonify(result.get("render_status_path")),
            "per_geometry_metrics": _jsonify(result.get("per_geometry_metrics")),
            "bootstrap_intervals": _jsonify(result.get("bootstrap_intervals")),
            "baseline_comparison": _jsonify(result.get("baseline_comparison")),
            "render_status": _jsonify(result.get("render_status")),
            "per_geometry_slices": _jsonify(result.get("per_geometry_slices")),
        },
    }
    launcher_summary_path = output_dir / "launcher_summary.json"
    launcher_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("standard_evaluation_done", flush=True)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    print(f"launcher_summary_path={launcher_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pqt.harness import RunSpec, WorkspacePaths, execute_run, resolve_run
from pqt.workspace import (
    WorkspaceLayout,
    claim_workspace,
    declare_connector,
    initialize_worktrees,
    release_workspace,
    synchronize_connector,
    workspace_status,
)


def _four_positive_ints(value: str) -> tuple[int, int, int, int]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected four comma-separated integers") from exc
    if len(parsed) != 4 or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected four positive comma-separated integers")
    return parsed


def _add_run_arguments(parser: argparse.ArgumentParser, *, checkpoint: bool = False, image_dir: bool = False) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--dataset-name", default="data1")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    if checkpoint:
        parser.add_argument("--checkpoint-path", type=Path)
    if image_dir:
        parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tgpnet", description="Explicit tgpnet run and workspace harness.")
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ("train", "evaluate", "predict", "diagnose", "verify"):
        child = commands.add_parser(mode)
        _add_run_arguments(child, checkpoint=mode in {"evaluate", "predict", "diagnose"}, image_dir=mode == "predict")
        if mode == "train":
            child.add_argument("--split-strategy", choices=("spatial_block", "region_block", "level_holdout", "full"))
            child.add_argument("--cv-fold", type=int)
            child.add_argument("--epochs", type=int)
            child.add_argument("--optimization-seed", type=int)
            child.add_argument("--shuffle-train-labels", action="store_true")
            child.add_argument("--train-contrast-factor", type=float)
            child.add_argument("--val-curve-warp", dest="val_curve_time_warp", type=float)
            child.add_argument("--diagnostics", action="store_true")
            child.add_argument("--diagnostic-epoch-interval", type=int, default=5)
            child.add_argument("--diagnostic-batches", type=int, default=1)
            child.add_argument("--diagnostic-samples", type=int, default=1)
            child.add_argument(
                "--backbone-name",
                choices=("tgp_pqt_convnext_v1", "efficientnet_v2_s", "resnet50", "tgp_thermo_seq_v1", "tgp_thermo_geo_v1"),
            )
            child.add_argument(
                "--head-variant",
                choices=("pqt_rrh", "pqt_linear"),
            )
            child.add_argument("--sra-encoding", action=argparse.BooleanOptionalAction, default=None)
            child.add_argument("--cross-scale-refinement", action=argparse.BooleanOptionalAction, default=None)
            child.add_argument("--se-refinement", action=argparse.BooleanOptionalAction, default=None)
            child.add_argument("--loss-name", choices=("mse", "huber"))
            child.add_argument("--huber-delta", type=float)
            child.add_argument("--quantile-tau", type=float)
            child.add_argument("--depth-weight-gamma", type=float)
            child.add_argument("--ordinal-loss-weight", type=float)
            child.add_argument("--lr", type=float)
            child.add_argument("--weight-decay", type=float)
            child.add_argument("--backbone-lr-multiplier", type=float)
            child.add_argument("--neck-lr-multiplier", type=float)
            child.add_argument("--head-lr-multiplier", type=float)
            child.add_argument("--clip-grad", action=argparse.BooleanOptionalAction, default=None)
            child.add_argument("--max-grad-norm", type=float)

    workspace = commands.add_parser("workspace")
    actions = workspace.add_subparsers(dest="action", required=True)
    actions.add_parser("init")
    actions.add_parser("status")
    claim = actions.add_parser("claim")
    claim.add_argument("slot", choices=("treeA", "treeB", "treeC", "treeD"))
    claim.add_argument("--owner", required=True)
    claim.add_argument("--task", required=True)
    claim.add_argument("--job-id")
    release = actions.add_parser("release")
    release.add_argument("slot", choices=("treeA", "treeB", "treeC", "treeD"))
    release.add_argument("--token", required=True)
    connect = actions.add_parser("connect")
    connect.add_argument("--source", required=True, choices=("treeA", "treeB", "treeC", "treeD"))
    connect.add_argument("--target", required=True, choices=("treeA", "treeB", "treeC", "treeD"))
    connect.add_argument("--source-token", required=True)
    connect.add_argument("--target-token", required=True)
    connect.add_argument("--purpose", required=True)
    sync = actions.add_parser("sync")
    sync.add_argument("connector_id")
    sync.add_argument("--source-token", required=True)
    sync.add_argument("--target-token", required=True)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    paths = WorkspacePaths.discover()
    if args.asset_root:
        paths = WorkspacePaths(paths.code_root, args.asset_root.resolve(), paths.run_root)
    if args.run_root:
        paths = WorkspacePaths(paths.code_root, paths.asset_root, args.run_root.resolve())
    overrides = {
        name: getattr(args, name, None)
        for name in (
            "device", "gpu_id", "batch_size", "num_workers", "optimization_seed",
            "gradient_accumulation_steps", "mixed_precision",
        )
        if getattr(args, name, None) is not None
    }
    spec = RunSpec(
        mode=args.command,
        dataset_name=args.dataset_name,
        experiment_id=args.experiment_id,
        checkpoint_path=getattr(args, "checkpoint_path", None),
        image_dir=getattr(args, "image_dir", None),
        run_id=args.run_id,
        runtime_overrides=overrides,
        split_strategy=getattr(args, "split_strategy", None),
        cv_fold=getattr(args, "cv_fold", None),
        epochs=getattr(args, "epochs", None),
        shuffle_train_labels=getattr(args, "shuffle_train_labels", False),
        model_overrides={
            name: getattr(args, name)
            for name in (
                "backbone_name", "head_variant", "sra_encoding",
                "cross_scale_refinement", "se_refinement",
            )
            if getattr(args, name, None) is not None
        },
        data_overrides={
            name: getattr(args, name)
            for name in ("train_contrast_factor", "val_curve_time_warp")
            if getattr(args, name, None) is not None
        } or None,
        optimization_overrides={
            name: getattr(args, name)
            for name in ("loss_name", "huber_delta", "quantile_tau", "depth_weight_gamma", "ordinal_loss_weight", "lr", "weight_decay", "backbone_lr_multiplier", "neck_lr_multiplier", "head_lr_multiplier", "clip_grad", "max_grad_norm")
            if getattr(args, name, None) is not None
        },
    )
    resolved = resolve_run(spec, paths)
    if args.command == "train" and args.diagnostics:
        resolved.config.diagnostics.enabled = True
        resolved.config.diagnostics.snapshot_interval = args.diagnostic_epoch_interval
        resolved.config.diagnostics.batches_per_snapshot = args.diagnostic_batches
        resolved.config.diagnostics.samples_per_snapshot = args.diagnostic_samples
    result = execute_run(resolved)
    print(json.dumps({"run_id": resolved.run_id, "run_dir": str(resolved.run_dir), "result": str(result)}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "workspace":
        return _run_command(args)
    layout = WorkspaceLayout.discover()
    if args.action == "init":
        result = initialize_worktrees(layout)
    elif args.action == "status":
        result = workspace_status(layout)
    elif args.action == "claim":
        result = claim_workspace(layout, args.slot, args.owner, args.task, args.job_id)
    elif args.action == "release":
        release_workspace(layout, args.slot, args.token)
        result = {"released": args.slot}
    elif args.action == "connect":
        result = declare_connector(
            layout, args.source, args.target, args.source_token, args.target_token, args.purpose
        )
    else:
        result = synchronize_connector(layout, args.connector_id, args.source_token, args.target_token)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

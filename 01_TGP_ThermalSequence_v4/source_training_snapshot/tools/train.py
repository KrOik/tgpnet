from __future__ import annotations

import argparse
import sys
from pathlib import Path


CANONICAL_EXPERIMENT_ID = (
    "backbone-convnext-stage-fuse-neck-physics-band-decouple-topology-conditioned-routing-loss-"
    "barron-adaptive-probe42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the canonical tgpnet model.")
    parser.add_argument("--dataset-name", default="data1")
    parser.add_argument("--experiment-id", default=CANONICAL_EXPERIMENT_ID)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-epoch-interval", type=int, default=5)
    parser.add_argument("--diagnostic-batches", type=int, default=1)
    parser.add_argument("--diagnostic-samples", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pqt.config.defaults import build_default_train_config
    from pqt.workflows.train import run_training

    config = build_default_train_config(dataset_name=args.dataset_name)
    config.experiment.experiment_id = args.experiment_id
    if args.output_dir:
        config.output.output_dir = Path(args.output_dir).resolve()
    if args.epochs is not None:
        config.optimization.epochs = args.epochs
    if args.batch_size is not None:
        config.runtime.batch_size = args.batch_size
    if args.num_workers is not None:
        config.runtime.num_workers = args.num_workers
    config.runtime.device = args.device
    config.runtime.gpu_id = args.gpu_id
    config.diagnostics.enabled = bool(args.diagnostics)
    config.diagnostics.epoch_interval = args.diagnostic_epoch_interval
    config.diagnostics.batches_per_snapshot = args.diagnostic_batches
    config.diagnostics.samples_per_snapshot = args.diagnostic_samples

    result = run_training(config=config)
    print(f"best_checkpoint={result['best_checkpoint_path']}")
    print(f"final_checkpoint={result['final_checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

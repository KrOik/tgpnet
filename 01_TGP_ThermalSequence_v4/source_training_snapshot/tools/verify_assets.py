from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify tgpnet datasets and checkpoints against manifests.")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from pqt.assets import verify_assets

    args = parse_args()
    errors = verify_assets(args.asset_root, args.manifest_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    import csv
    with (args.manifest_root / "manifests" / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        dataset_count = sum(1 for _ in csv.DictReader(handle))
    with (args.manifest_root / "manifests" / "models.csv").open(newline="", encoding="utf-8") as handle:
        model_count = sum(1 for _ in csv.DictReader(handle))
    print(f"verified {dataset_count} datasets and {model_count} model assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

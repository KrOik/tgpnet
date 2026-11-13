from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a fixed final epoch from four spatial-validation histories.")
    parser.add_argument("histories", nargs=4, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_epoch(histories: list[Path]) -> dict[str, object]:
    folds = []
    best_epochs = []
    for fold, path in enumerate(histories):
        audit_path = path.parent / "split_audit.json"
        if not audit_path.is_file():
            raise ValueError(f"Missing split audit beside history: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_audit = {
            "strategy": "spatial_block",
            "fold": fold,
            "num_folds": 4,
            "guard_radius": 0,
            "seed": 42,
            "shuffle_train_labels_negative_control": False,
            "test_sample_size": 0,
            "external_data_access": False,
        }
        mismatches = {
            key: {"expected": value, "actual": audit.get(key)}
            for key, value in expected_audit.items()
            if audit.get(key) != value
        }
        if mismatches:
            raise ValueError(f"History is not from the preregistered spatial fold {fold}: {mismatches}")
        if int(audit.get("train_sample_size", 0)) <= 0 or int(audit.get("validation_sample_size", 0)) <= 0:
            raise ValueError(f"Spatial fold {fold} has empty train or validation data.")
        frame = pd.read_csv(path)
        if "val_mae" not in frame or not frame["val_mae"].notna().any():
            raise ValueError(f"History has no spatial validation MAE: {path}")
        index = int(frame["val_mae"].astype(float).idxmin())
        epoch = index + 1
        best_epochs.append(epoch)
        folds.append(
            {
                "fold": fold,
                "history": str(path),
                "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split_audit": str(audit_path),
                "split_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "best_epoch": epoch,
                "best_val_mae": float(frame.loc[index, "val_mae"]),
            }
        )
    median = float(np.median(best_epochs))
    selected = int(math.floor(median + 0.5))
    return {
        "protocol": "four-fold-spatial-block-median-epoch-v1",
        "folds": folds,
        "best_epochs": best_epochs,
        "median_best_epoch": median,
        "selected_final_epoch": selected,
    }


def main() -> int:
    args = parse_args()
    result = select_epoch(args.histories)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

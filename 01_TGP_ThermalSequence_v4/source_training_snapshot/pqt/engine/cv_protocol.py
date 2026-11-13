from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from pqt.engine.candidate_selection import half_up_median_epoch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_spatial_cv(histories: list[Path]) -> dict[str, object]:
    if len(histories) != 4:
        raise ValueError("Spatial CV summary requires exactly four histories.")
    folds: list[dict[str, object]] = []
    for expected_fold, history_path in enumerate(histories):
        history_path = Path(history_path).resolve()
        audit_path = history_path.parent / "split_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        required = {
            "strategy": "spatial_block", "fold": expected_fold, "num_folds": 4,
            "guard_radius": 0, "seed": 42, "external_data_access": False,
        }
        if any(audit.get(name) != value for name, value in required.items()):
            raise ValueError(f"History is not from the preregistered fold {expected_fold}: {history_path}")
        frame = pd.read_csv(history_path)
        required_columns = {"val_mae", "val_rmse", "val_bias", "val_mae_shallow", "val_mae_middle", "val_mae_deep"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"History lacks candidate-selection metrics: {history_path}")
        best_index = int(frame["val_mae"].astype(float).idxmin())
        row = frame.loc[best_index]
        folds.append({
            "fold": expected_fold, "best_epoch": best_index + 1,
            **{name: float(row[name]) for name in required_columns},
            "history": str(history_path), "history_sha256": _sha256(history_path),
            "split_audit": str(audit_path), "split_audit_sha256": _sha256(audit_path),
            "depth_max": audit.get("depth_max"),
        })
    metrics = ("val_mae", "val_rmse", "val_bias", "val_mae_shallow", "val_mae_middle", "val_mae_deep")
    return {
        "protocol": "four-fold-spatial-block-candidate-v1",
        "folds": folds,
        "mean": {name: float(sum(float(fold[name]) for fold in folds) / 4.0) for name in metrics},
        "best_epochs": [int(fold["best_epoch"]) for fold in folds],
        "selected_final_epoch": half_up_median_epoch(int(fold["best_epoch"]) for fold in folds),
    }

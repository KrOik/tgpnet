from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pqt.config.schema import SplitConfig


@dataclass(slots=True)
class SplitDataFrameBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_dataframe(dataframe: pd.DataFrame, split_config: SplitConfig) -> SplitDataFrameBundle:
    if split_config.strategy == "full":
        empty = dataframe.iloc[0:0].copy().reset_index(drop=True)
        return SplitDataFrameBundle(train=dataframe.copy().reset_index(drop=True), val=empty, test=empty.copy())
    if split_config.strategy == "spatial_block":
        return _split_spatial_blocks(dataframe, split_config)
    if split_config.strategy == "region_block":
        return _split_global_regions(dataframe, split_config)
    if split_config.strategy == "level_holdout":
        return _split_level_holdout(dataframe, split_config)

    # Keep the split order aligned with the legacy baseline implementation.
    shuffled = dataframe.sample(frac=1, random_state=split_config.seed).reset_index(drop=True)

    total_size = len(shuffled)
    train_size = int(total_size * split_config.train_ratio)
    val_size = int(total_size * split_config.val_ratio)

    train = shuffled.iloc[:train_size].reset_index(drop=True)
    val = shuffled.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test = shuffled.iloc[train_size + val_size :].reset_index(drop=True)
    return SplitDataFrameBundle(train=train, val=val, test=test)


def _split_spatial_blocks(dataframe: pd.DataFrame, split_config: SplitConfig) -> SplitDataFrameBundle:
    coordinates = _parse_coordinates(dataframe)

    working = dataframe.copy()
    working["__x"] = coordinates["X"].astype(int)
    working["__y"] = coordinates["Y"].astype(int)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for _, group in working.groupby("Label", sort=True):
        center_x = float(group["__x"].median())
        center_y = float(group["__y"].median())
        quadrants = (group["__x"] >= center_x).astype(int) + 2 * (group["__y"] >= center_y).astype(int)
        validation = group.loc[quadrants == split_config.fold]
        candidates = group.loc[quadrants != split_config.fold]
        if validation.empty or candidates.empty:
            raise ValueError(f"Depth {group['Label'].iloc[0]} does not populate every required spatial side.")

        validation_xy = validation[["__x", "__y"]].to_numpy()
        candidate_xy = candidates[["__x", "__y"]].to_numpy()
        separated = []
        for xy in candidate_xy:
            chebyshev = abs(validation_xy - xy).max(axis=1)
            separated.append(bool((chebyshev > split_config.guard_radius).all()))
        training = candidates.loc[separated]
        if training.empty:
            raise ValueError(f"Guard radius removes every training point for depth {group['Label'].iloc[0]}.")
        train_indices.extend(training.index.tolist())
        validation_indices.extend(validation.index.tolist())

    train = dataframe.loc[train_indices].sort_index().reset_index(drop=True)
    val = dataframe.loc[validation_indices].sort_index().reset_index(drop=True)
    empty = dataframe.iloc[0:0].copy().reset_index(drop=True)
    expected_labels = set(dataframe["Label"].astype(float))
    if set(train["Label"].astype(float)) != expected_labels or set(val["Label"].astype(float)) != expected_labels:
        raise RuntimeError("Every spatial fold must retain all depth levels in train and validation.")
    return SplitDataFrameBundle(train=train, val=val, test=empty)


def _split_global_regions(dataframe: pd.DataFrame, split_config: SplitConfig) -> SplitDataFrameBundle:
    coordinates = _parse_coordinates(dataframe)
    working = dataframe.copy()
    working["__x"] = coordinates["X"].astype(int)
    working["__y"] = coordinates["Y"].astype(int)

    center_x = float(working["__x"].median())
    center_y = float(working["__y"].median())
    quadrants = (working["__x"] >= center_x).astype(int) + 2 * (working["__y"] >= center_y).astype(int)
    validation = working.loc[quadrants == split_config.fold]
    candidates = working.loc[quadrants != split_config.fold]
    if validation.empty or candidates.empty:
        raise ValueError(f"Global region fold {split_config.fold} is empty.")

    validation_xy = validation[["__x", "__y"]].to_numpy()
    candidate_xy = candidates[["__x", "__y"]].to_numpy()
    separated = [
        bool((abs(validation_xy - xy).max(axis=1) > split_config.guard_radius).all())
        for xy in candidate_xy
    ]
    training = candidates.loc[separated]
    if training.empty:
        raise ValueError("Guard radius removes every global-region training point.")

    columns = dataframe.columns
    train = training.loc[:, columns].sort_index().reset_index(drop=True)
    val = validation.loc[:, columns].sort_index().reset_index(drop=True)
    empty = dataframe.iloc[0:0].copy().reset_index(drop=True)
    return SplitDataFrameBundle(train=train, val=val, test=empty)


def _split_level_holdout(dataframe: pd.DataFrame, split_config: SplitConfig) -> SplitDataFrameBundle:
    """Leave-one-depth-level-out: fold f holds out the f-th sorted unique label.

    Interior-level folds probe interpolation onto unseen label values, which is
    what external continuous-band labels require. Edge folds (first/last level)
    are extrapolation probes.
    """
    levels = sorted(dataframe["Label"].astype(float).unique().tolist())
    if split_config.fold >= len(levels):
        raise ValueError(f"level_holdout fold {split_config.fold} out of range for {len(levels)} levels.")
    held = levels[split_config.fold]
    validation_mask = dataframe["Label"].astype(float) == held
    val = dataframe.loc[validation_mask].sort_index().reset_index(drop=True)
    train = dataframe.loc[~validation_mask].sort_index().reset_index(drop=True)
    if val.empty or train.empty:
        raise ValueError(f"level_holdout fold {split_config.fold} (level {held}) produced an empty side.")
    if held in set(train["Label"].astype(float)):
        raise RuntimeError("level_holdout leaked the held-out level into training.")
    empty = dataframe.iloc[0:0].copy().reset_index(drop=True)
    return SplitDataFrameBundle(train=train, val=val, test=empty)


def _parse_coordinates(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"Image_Path", "Label"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Spatial split requires columns {sorted(required)}, missing {sorted(missing)}.")
    coordinates = dataframe["Image_Path"].astype(str).str.extract(r"^(?P<X>-?\d+)_(?P<Y>-?\d+)\.[^.]+$")
    if coordinates.isna().any().any():
        invalid = dataframe.loc[coordinates.isna().any(axis=1), "Image_Path"].astype(str).head(5).tolist()
        raise ValueError(f"Spatial split cannot parse coordinates from filenames: {invalid}")
    return coordinates


def export_split_csvs(bundle: SplitDataFrameBundle, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / f"{stem}_train.csv",
        "val": output_dir / f"{stem}_val.csv",
        "test": output_dir / f"{stem}_test.csv",
    }
    bundle.train.to_csv(paths["train"], index=False)
    bundle.val.to_csv(paths["val"], index=False)
    bundle.test.to_csv(paths["test"], index=False)
    return paths

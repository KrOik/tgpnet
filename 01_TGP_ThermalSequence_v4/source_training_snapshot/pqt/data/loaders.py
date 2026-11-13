from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from pqt.config.schema import DataConfig, RuntimeConfig, SplitConfig

from .datasets import CsvImageRegressionDataset, ImageFolderDataset
from .splits import export_split_csvs, load_dataframe, split_dataframe
from .transforms import build_transform
from .sr_cache import verify_sr_cache


@dataclass(slots=True)
class DataLoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _pin_memory_enabled(runtime_config: RuntimeConfig) -> bool:
    return runtime_config.pin_memory and runtime_config.device != "cpu"


def create_supervised_dataloaders(
    data_config: DataConfig,
    split_config: SplitConfig,
    runtime_config: RuntimeConfig,
) -> DataLoaderBundle:
    if data_config.csv_path is None or data_config.image_dir is None:
        raise ValueError("Supervised dataloaders require csv_path and image_dir.")

    train_transform = build_transform(
        data_config.input_mode,
        data_config.image_size,
        train_contrast_factor=data_config.train_contrast_factor,
    )
    evaluation_transform = build_transform(data_config.input_mode, data_config.image_size)
    dataframe = load_dataframe(data_config.csv_path)
    requested_names = set(dataframe[data_config.filename_column].astype(str))
    if data_config.sr_image_dir is not None:
        if data_config.sr_manifest_path is None:
            raise ValueError("sr_manifest_path is required with sr_image_dir.")
        verify_sr_cache(data_config.sr_image_dir, data_config.sr_manifest_path, data_config.image_dir, requested_names)
    splits = split_dataframe(dataframe, split_config)
    if data_config.shuffle_train_labels:
        labels = np.sort(splits.train[data_config.label_column].astype(float).unique())
        if len(labels) < 2:
            raise ValueError("Label-shuffle control requires at least two depth groups.")
        rng = np.random.default_rng(split_config.seed)
        shuffled = rng.permutation(labels)
        while np.any(shuffled == labels):
            shuffled = rng.permutation(labels)
        mapping = dict(zip(labels.tolist(), shuffled.tolist(), strict=True))
        splits.train.loc[:, data_config.label_column] = (
            splits.train[data_config.label_column].astype(float).map(mapping)
        )

    if data_config.export_splits:
        output_dir = data_config.split_output_dir or data_config.csv_path.parent
        export_split_csvs(splits, output_dir, data_config.csv_path.stem)

    if data_config.input_mode == "curve":
        from .curve_cache import load_or_build_curve_cache
        from .datasets import CsvCurveRegressionDataset

        if data_config.context_neighbor_count > 0:
            raise NotImplementedError("curve input mode does not support neighborhood context yet.")
        curve_cache = load_or_build_curve_cache(data_config.image_dir)
        train_curve_transform = None
        if data_config.train_curve_augment > 0:
            from .transforms import CurveAugment

            train_curve_transform = CurveAugment(data_config.train_curve_augment)
        train_dataset = CsvCurveRegressionDataset(
            dataframe=splits.train,
            cache=curve_cache,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
            transform=train_curve_transform,
            time_warp=data_config.train_curve_time_warp,
        )
        val_dataset = CsvCurveRegressionDataset(
            dataframe=splits.val,
            cache=curve_cache,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
            val_time_warp=data_config.val_curve_time_warp,
        )
        test_dataset = CsvCurveRegressionDataset(
            dataframe=splits.test,
            cache=curve_cache,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
        )
    else:
        train_dataset = CsvImageRegressionDataset(
            dataframe=splits.train,
            image_dir=data_config.image_dir,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
            transform=train_transform,
            input_mode=data_config.input_mode,
            sr_image_dir=data_config.sr_image_dir,
        )
        val_dataset = CsvImageRegressionDataset(
            dataframe=splits.val,
            image_dir=data_config.image_dir,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
            transform=evaluation_transform,
            input_mode=data_config.input_mode,
            sr_image_dir=data_config.sr_image_dir,
        )
        test_dataset = CsvImageRegressionDataset(
            dataframe=splits.test,
            image_dir=data_config.image_dir,
            filename_column=data_config.filename_column,
            label_column=data_config.label_column,
            transform=evaluation_transform,
            input_mode=data_config.input_mode,
            sr_image_dir=data_config.sr_image_dir,
        )

    if data_config.context_neighbor_count > 0:
        from .neighborhood import NeighborContextDataset

        train_dataset, val_dataset, test_dataset = (
            NeighborContextDataset(
                dataset,
                image_dir=data_config.image_dir,
                filename_column=data_config.filename_column,
                neighbor_count=data_config.context_neighbor_count,
                radius=data_config.context_radius,
                transform=evaluation_transform,
                input_mode=data_config.input_mode,
            )
            for dataset in (train_dataset, val_dataset, test_dataset)
        )

    effective_optimization_seed = runtime_config.effective_optimization_seed(split_config.seed)
    train_generator = torch.Generator().manual_seed(effective_optimization_seed)
    return DataLoaderBundle(
        train=DataLoader(
            train_dataset,
            batch_size=runtime_config.batch_size,
            shuffle=True,
            num_workers=runtime_config.num_workers,
            pin_memory=_pin_memory_enabled(runtime_config),
            worker_init_fn=_seed_worker,
            generator=train_generator,
        ),
        val=DataLoader(
            val_dataset,
            batch_size=runtime_config.batch_size,
            shuffle=False,
            num_workers=runtime_config.num_workers,
            pin_memory=_pin_memory_enabled(runtime_config),
            worker_init_fn=_seed_worker,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=runtime_config.batch_size,
            shuffle=False,
            num_workers=runtime_config.num_workers,
            pin_memory=_pin_memory_enabled(runtime_config),
            worker_init_fn=_seed_worker,
        ),
    )


def create_image_prediction_loader(data_config: DataConfig, runtime_config: RuntimeConfig) -> DataLoader:
    if data_config.image_dir is None:
        raise ValueError("Prediction loader requires image_dir.")

    transform = build_transform(data_config.input_mode, data_config.image_size)
    if data_config.sr_image_dir is not None:
        if data_config.sr_manifest_path is None:
            raise ValueError("sr_manifest_path is required with sr_image_dir.")
        verify_sr_cache(data_config.sr_image_dir, data_config.sr_manifest_path, data_config.image_dir)
    dataset = ImageFolderDataset(
        image_dir=data_config.image_dir,
        valid_extensions=data_config.valid_extensions,
        transform=transform,
        input_mode=data_config.input_mode,
        sr_image_dir=data_config.sr_image_dir,
    )
    if data_config.context_neighbor_count > 0:
        from .neighborhood import FolderNeighborContextDataset

        dataset = FolderNeighborContextDataset(
            dataset,
            image_dir=data_config.image_dir,
            neighbor_count=data_config.context_neighbor_count,
            radius=data_config.context_radius,
            transform=transform,
            input_mode=data_config.input_mode,
        )
    return DataLoader(
        dataset,
        batch_size=runtime_config.batch_size,
        shuffle=False,
        num_workers=runtime_config.num_workers,
        pin_memory=_pin_memory_enabled(runtime_config),
    )

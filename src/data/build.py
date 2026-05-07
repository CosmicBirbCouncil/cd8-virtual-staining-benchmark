from __future__ import annotations

from hydra.utils import instantiate
from typing import Any, Callable

from src.data.transforms.factory import build_transform
from src.data.datasets.paired import PairedDiffusionDataset


def build_paired_dataset(cfg, split: str) -> PairedDiffusionDataset:
    transform = build_transform(cfg, split)

    if split == "train":
        data_path = cfg.data.paths.train_path
    elif split == "val":
        data_path = cfg.data.paths.valid_path
    elif split == "test":
        data_path = cfg.data.paths.valid_path
    else:
        raise ValueError("Incorrect data split provided. Should be train, val, or test. ")

    return PairedDiffusionDataset(
        manifest_path=data_path,
        split=split,
        transform=transform
    )

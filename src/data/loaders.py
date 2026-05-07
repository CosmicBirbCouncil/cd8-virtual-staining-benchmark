from __future__ import annotations
from typing import Any, Callable

from torch.utils.data import DataLoader, DistributedSampler

from src.data.build import build_paired_dataset


def build_dataloader(cfg, split: str) -> DataLoader:
    dataset = build_paired_dataset(cfg, split=split)
    is_train = split == "train"
    num_workers = int(cfg.data.loader.num_workers)

    if cfg.trainer.distributed or cfg.generate.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(
            dataset,
            batch_size=int(cfg.data.loader.batch_size),
            sampler=dist_sampler,
            num_workers=num_workers,
            pin_memory=bool(cfg.data.loader.pin_memory),
        )
    

    return DataLoader(
        dataset,
        batch_size=int(cfg.data.loader.batch_size),
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.loader.pin_memory),
    )

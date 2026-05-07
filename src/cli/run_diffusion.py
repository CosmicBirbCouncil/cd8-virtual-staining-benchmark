from __future__ import annotations

import logging
from pathlib import Path


import hydra
import os
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.trainer.diffusion import DiffusionTrainer
from src.trainer.dab import DiffusionTrainerWithDABLoss, DiffusionTrainerWithDABLossEMA

log = logging.getLogger(__name__)


def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


import os
import torch
import torch.distributed as dist


def setup_distributed(cfg):
    if not getattr(cfg.trainer, "distributed", False):
        return 0, 1, False

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    return local_rank, world_size, True


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger(getattr(cfg, "log_level", "INFO"))
    log.info("Diffusion training config:\n%s", OmegaConf.to_yaml(cfg))
    local_rank, world_size, is_distributed = setup_distributed(cfg)

    # diffusion_trainer = DiffusionTrainer(cfg)
    diffusion_trainer = DiffusionTrainerWithDABLossEMA(cfg)
    diffusion_trainer.train()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    
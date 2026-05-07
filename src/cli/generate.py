from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.build import build_model, build_scheduler
from src.data.loaders import build_dataloader


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(cfg: DictConfig) -> torch.device:
    if getattr(cfg.trainer, "distributed", False) and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device(f"cuda:{local_rank}")
    return torch.device(cfg.generate.device if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def denormalize_tile(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    arr = t.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


@torch.no_grad()
def generate_batch(
    model,
    scheduler,
    condition: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    condition: [B, C, H, W]
    returns:   [B, C, H, W]
    """
    condition = condition.to(device)
    x = torch.randn_like(condition, device=device)

    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in scheduler.timesteps:
        timesteps = torch.full(
            (x.shape[0],),
            t,
            device=device,
            dtype=torch.long,
        )
        model_input = torch.cat([x, condition], dim=1)
        noise_pred = model(model_input, timesteps).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x


def _to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x


def _get_batch_item(batch_value, idx):
    """
    Safely get an item from batched metadata.
    """
    if isinstance(batch_value, torch.Tensor):
        item = batch_value[idx]
        return _to_python_scalar(item)
    if isinstance(batch_value, (list, tuple)):
        return batch_value[idx]
    return batch_value


def run_generate(cfg: DictConfig) -> None:
    device = setup_distributed(cfg)

    output_dir = Path(cfg.generate.output_dir)
    output_manifest_csv = Path(cfg.generate.output_manifest_csv)
    checkpoint_path = Path(cfg.generate.checkpoint_path)
    num_inference_steps = int(cfg.generate.num_inference_steps)
    split = str(cfg.generate.split)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    scheduler = build_scheduler(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Use your existing dataloader setup; it should already shard in distributed mode
    loader = build_dataloader(cfg, split)

    rows = []

    progress = tqdm(
        loader,
        desc=f"Generating [rank {get_rank()}]",
        disable=not is_main_process(),
    )

    for batch in progress:
        condition = batch["condition_image"].to(device)

        fake_batch = generate_batch(
            model=model,
            scheduler=scheduler,
            condition=condition,
            num_inference_steps=num_inference_steps,
            device=device,
        )

        bsz = fake_batch.shape[0]

        for i in range(bsz):
            try:
                fake_arr = denormalize_tile(fake_batch[i])

                case_id = (
                    _get_batch_item(batch["case_id"], i)
                    if "case_id" in batch
                    else "unknown"
                )
                tile_id = (
                    _get_batch_item(batch["tile_id"], i)
                    if "tile_id" in batch
                    else f"sample_{i}"
                )

                save_dir = output_dir / str(case_id)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{tile_id}.npy"
                np.save(save_path, fake_arr)

                out_row = {
                    "case_id": case_id,
                    "tile_id": tile_id,
                    "generated_tile": str(save_path),
                }

                # Preserve useful metadata if present
                for key in ["input_tile", "target_tile", "x", "y", "level", "split"]:
                    if key in batch:
                        out_row[key] = _get_batch_item(batch[key], i)

                rows.append(out_row)

            except Exception as e:
                out_row = {
                    "case_id": _get_batch_item(batch["case_id"], i) if "case_id" in batch else "unknown",
                    "tile_id": _get_batch_item(batch["tile_id"], i) if "tile_id" in batch else f"sample_{i}",
                    "generated_tile": np.nan,
                    "generation_error": str(e),
                }

                for key in ["input_tile", "target_tile", "x", "y", "level", "split"]:
                    if key in batch:
                        out_row[key] = _get_batch_item(batch[key], i)

                rows.append(out_row)

    # Each rank writes its own partial CSV
    rank = get_rank()
    partial_csv = output_manifest_csv.parent / f"{output_manifest_csv.stem}_rank{rank}{output_manifest_csv.suffix}"
    pd.DataFrame(rows).to_csv(partial_csv, index=False)

    if is_distributed():
        dist.barrier()

    # Rank 0 merges all partial manifests
    if is_main_process():
        partials = []
        for r in range(get_world_size()):
            p = output_manifest_csv.parent / f"{output_manifest_csv.stem}_rank{r}{output_manifest_csv.suffix}"
            if p.exists():
                partials.append(pd.read_csv(p))

        if len(partials) == 0:
            raise RuntimeError("No partial generation manifests found.")

        merged = pd.concat(partials, ignore_index=True)
        merged.to_csv(output_manifest_csv, index=False)

        print(f"Saved generated tiles to {output_dir}")
        print(f"Saved merged generation manifest to {output_manifest_csv}")

    cleanup_distributed()


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if is_main_process():
        print(OmegaConf.to_yaml(cfg))
    run_generate(cfg)


if __name__ == "__main__":
    main()
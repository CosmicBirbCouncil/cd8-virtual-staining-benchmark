from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from hydra.utils import instantiate
from tqdm import tqdm

from src.data.loaders import build_dataloader
from src.models.build import build_model, build_scheduler, build_optimizer
from src.logging.wandb import WandbLogger


class DiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        if getattr(cfg.trainer, "distributed", False) and torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                cfg.trainer.device if torch.cuda.is_available() else "cpu"
            )

        self.model = build_model(cfg).to(self.device)
        self.noise_scheduler = build_scheduler(cfg)
        self.num_inference_steps = cfg.scheduler.num_inference_timesteps
        self.optimizer = build_optimizer(cfg, self.model)
        self.criterion = instantiate(cfg.loss)
        self.train_loader = build_dataloader(cfg, "train")
        self.val_loader = build_dataloader(cfg, "val")

        self.ckpt_dir = Path(cfg.trainer.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = self.ckpt_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.best_val = float("inf")

        self.use_amp = bool(cfg.trainer.precision.mixed) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # W&B
        self.wandb = WandbLogger(
            cfg=cfg,
            enabled=getattr(cfg.logging, "use_wandb", False),
            is_main_process=self.is_main_process(),
        )
        self.wandb.set_model(self.model)
        self.wandb.init()

    def is_main_process(self):
        return (
            (not dist.is_available())
            or (not dist.is_initialized())
            or dist.get_rank() == 0
        )

    def forward_step(self, batch):
        condition = batch["condition_image"].to(self.device)
        target = batch["target_image"].to(self.device)

        bsz = target.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(target)
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)

        model_input = torch.cat([noisy_target, condition], dim=1)
        pred = self.model(model_input, timesteps).sample

        if self.noise_scheduler.prediction_type == "v_prediction":
            velocity = self.noise_scheduler.get_velocity(target, noise, timesteps)
            loss = self.criterion(pred, velocity)
        else:
            loss = self.criterion(pred, noise)

        return loss

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.is_main_process(),
        )

        for batch in progress:
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss = self.forward_step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.forward_step(batch)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            disable=not self.is_main_process(),
        )

        for batch in progress:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss = self.forward_step(batch)
            else:
                loss = self.forward_step(batch)

            total_loss += loss.item()

        return total_loss / max(1, len(self.val_loader))

    @torch.no_grad()
    def visualize_samples(self, epoch: int, num_samples: int = 4):
        self.model.eval()

        batch = next(iter(self.val_loader))
        condition = batch["condition_image"][:num_samples].to(self.device)
        target = batch["target_image"][:num_samples].to(self.device)

        x = torch.randn_like(target)

        self.noise_scheduler.set_timesteps(
            self.num_inference_steps,
            device=self.device,
        )

        for t in tqdm(
            self.noise_scheduler.timesteps,
            desc="Generating samples",
            disable=not self.is_main_process(),
        ):
            timesteps = torch.full(
                (x.shape[0],),
                t,
                device=self.device,
                dtype=torch.long,
            )
            model_input = torch.cat([x, condition], dim=1)
            noise_pred = self.model(model_input, timesteps).sample
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

        generated = x

        def normalize(img):
            return (img.clamp(-1, 1) + 1) / 2

        condition = normalize(condition)
        generated = normalize(generated)
        target = normalize(target)

        # Row order: HE | GEN | GT
        grid = torch.cat([condition, generated, target], dim=0)

        save_path = self.vis_dir / f"epoch_{epoch:03d}.png"
        vutils.save_image(grid, save_path, nrow=num_samples)

        print(f"[Visualization] Saved to {save_path}")
        return save_path

    def save_checkpoint(self, name: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.ckpt_dir / name,
        )

    def train(self):
        try:
            for epoch in range(1, int(self.cfg.trainer.epochs) + 1):
                if self.cfg.trainer.distributed and hasattr(self.train_loader, "sampler"):
                    self.train_loader.sampler.set_epoch(epoch)

                train_loss = self.train_one_epoch()
                val_loss = self.validate()

                print(
                    f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )

                viz_path = None
                if self.is_main_process():
                    viz_path = self.visualize_samples(epoch)

                if self.cfg.trainer.checkpoint.save_last:
                    self.save_checkpoint("last.pt", epoch)

                if self.cfg.trainer.checkpoint.save_best and val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint("best.pt", epoch)

                # W&B logging (rank 0 only handled by logger)
                self.wandb.log_metrics(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "best/val_loss": self.best_val,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

                if viz_path is not None:
                    self.wandb.log_image(
                        "samples/fixed_panel",
                        viz_path,
                        step=epoch,
                    )

        finally:
            self.wandb.finish()
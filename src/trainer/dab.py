from __future__ import annotations

import torch
from tqdm import tqdm

from src.trainer.diffusion import DiffusionTrainer
from src.losses.dab import DABLoss


class DiffusionTrainerWithDABLoss(DiffusionTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.lambda_dab = float(getattr(cfg.trainer, "lambda_dab", 0.0))
        self.dab_loss_fn = DABLoss(
            pos_threshold=float(getattr(cfg.trainer, "dab_pos_threshold", 0.05)),
            sharpness=float(getattr(cfg.trainer, "dab_sharpness", 40.0)),
            use_hist=bool(getattr(cfg.trainer, "dab_use_hist", True)),
            hist_bins=int(getattr(cfg.trainer, "dab_hist_bins", 16)),
        )

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct x0 directly from x_t and model output at training timesteps.
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_t = 1.0 - alpha_t

        if self.noise_scheduler.prediction_type == "epsilon":
            pred_x0 = (x_t - beta_t.sqrt() * model_pred) / alpha_t.sqrt()
        elif self.noise_scheduler.prediction_type == "v_prediction":
            pred_x0 = alpha_t.sqrt() * x_t - beta_t.sqrt() * model_pred
        else:
            raise ValueError(
                f"Unsupported prediction type: {self.noise_scheduler.prediction_type}"
            )

        return pred_x0.clamp(-1.0, 1.0)

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
            diffusion_loss = self.criterion(pred, velocity)
        else:
            diffusion_loss = self.criterion(pred, noise)

        if self.lambda_dab > 0.0:
            pred_x0 = self._predict_x0(
                x_t=noisy_target.float(),
                model_pred=pred.float(),
                timesteps=timesteps,
            )
            dab_loss, dab_stats = self.dab_loss_fn(
                fake_img=pred_x0.float(),
                real_img=target.float(),
            )
        else:
            dab_loss = diffusion_loss.new_tensor(0.0)
            dab_stats = {
                "fake_dab_mean": diffusion_loss.new_tensor(0.0),
                "real_dab_mean": diffusion_loss.new_tensor(0.0),
                "fake_pos_frac": diffusion_loss.new_tensor(0.0),
                "real_pos_frac": diffusion_loss.new_tensor(0.0),
                "dab_loss_mean": diffusion_loss.new_tensor(0.0),
                "dab_loss_pos": diffusion_loss.new_tensor(0.0),
                "dab_loss_hist": diffusion_loss.new_tensor(0.0),
            }

        total_loss = diffusion_loss.float() + self.lambda_dab * dab_loss.float()

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss.detach(),
            "dab_loss": dab_loss.detach(),
            **dab_stats,
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_dab_loss = 0.0
        total_fake_dab_mean = 0.0
        total_real_dab_mean = 0.0
        total_fake_pos_frac = 0.0
        total_real_pos_frac = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.is_main_process(),
        )

        for batch in progress:
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.forward_step(batch)
                loss = outputs["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.forward_step(batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

            total_loss += outputs["loss"].item()
            total_diffusion_loss += outputs["diffusion_loss"].item()
            total_dab_loss += outputs["dab_loss"].item()
            total_fake_dab_mean += outputs["fake_dab_mean"].item()
            total_real_dab_mean += outputs["real_dab_mean"].item()
            total_fake_pos_frac += outputs["fake_pos_frac"].item()
            total_real_pos_frac += outputs["real_pos_frac"].item()

        denom = max(1, len(self.train_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "dab_loss": total_dab_loss / denom,
            "fake_dab_mean": total_fake_dab_mean / denom,
            "real_dab_mean": total_real_dab_mean / denom,
            "fake_pos_frac": total_fake_pos_frac / denom,
            "real_pos_frac": total_real_pos_frac / denom,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_dab_loss = 0.0
        total_fake_dab_mean = 0.0
        total_real_dab_mean = 0.0
        total_fake_pos_frac = 0.0
        total_real_pos_frac = 0.0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            disable=not self.is_main_process(),
        )

        for batch in progress:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.forward_step(batch)
            else:
                outputs = self.forward_step(batch)

            total_loss += outputs["loss"].item()
            total_diffusion_loss += outputs["diffusion_loss"].item()
            total_dab_loss += outputs["dab_loss"].item()
            total_fake_dab_mean += outputs["fake_dab_mean"].item()
            total_real_dab_mean += outputs["real_dab_mean"].item()
            total_fake_pos_frac += outputs["fake_pos_frac"].item()
            total_real_pos_frac += outputs["real_pos_frac"].item()

        denom = max(1, len(self.val_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "dab_loss": total_dab_loss / denom,
            "fake_dab_mean": total_fake_dab_mean / denom,
            "real_dab_mean": total_real_dab_mean / denom,
            "fake_pos_frac": total_fake_pos_frac / denom,
            "real_pos_frac": total_real_pos_frac / denom,
        }

    def train(self):
        try:
            for epoch in range(1, int(self.cfg.trainer.epochs) + 1):
                if self.cfg.trainer.distributed and hasattr(self.train_loader, "sampler"):
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics = self.train_one_epoch()
                val_metrics = self.validate()

                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]

                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.4f} "
                    f"train_diff={train_metrics['diffusion_loss']:.4f} "
                    f"train_dab={train_metrics['dab_loss']:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_diff={val_metrics['diffusion_loss']:.4f} "
                    f"val_dab={val_metrics['dab_loss']:.4f}"
                )

                viz_path = None
                if self.is_main_process():
                    viz_path = self.visualize_samples(epoch)

                if self.cfg.trainer.checkpoint.save_last:
                    self.save_checkpoint("last.pt", epoch)

                if self.cfg.trainer.checkpoint.save_best and val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint("best.pt", epoch)

                self.wandb.log_metrics(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/diffusion_loss": train_metrics["diffusion_loss"],
                        "train/dab_loss": train_metrics["dab_loss"],
                        "train/fake_dab_mean": train_metrics["fake_dab_mean"],
                        "train/real_dab_mean": train_metrics["real_dab_mean"],
                        "train/fake_pos_frac": train_metrics["fake_pos_frac"],
                        "train/real_pos_frac": train_metrics["real_pos_frac"],
                        "val/loss": val_metrics["loss"],
                        "val/diffusion_loss": val_metrics["diffusion_loss"],
                        "val/dab_loss": val_metrics["dab_loss"],
                        "val/fake_dab_mean": val_metrics["fake_dab_mean"],
                        "val/real_dab_mean": val_metrics["real_dab_mean"],
                        "val/fake_pos_frac": val_metrics["fake_pos_frac"],
                        "val/real_pos_frac": val_metrics["real_pos_frac"],
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


import copy
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm


class DiffusionTrainerWithDABLossEMA(DiffusionTrainerWithDABLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

        # -------------------------
        # EMA
        # -------------------------
        self.use_ema = bool(getattr(cfg.trainer, "use_ema", True))
        self.ema_decay = float(getattr(cfg.trainer, "ema_decay", 0.999))

        if self.use_ema:
            self.ema_model = AveragedModel(
                self.model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                use_buffers=True,
            ).to(self.device)
        else:
            self.ema_model = None

        # -------------------------
        # LR scheduler
        # -------------------------
        self.lr_scheduler = None
        if hasattr(cfg.trainer, "lr_scheduler"):
            sched_cfg = cfg.trainer.lr_scheduler
            sched_type = getattr(sched_cfg, "type", None)

            if sched_type == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(sched_cfg.t_max),
                    eta_min=float(getattr(sched_cfg, "eta_min", 0.0)),
                )
            elif sched_type == "step":
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=int(sched_cfg.step_size),
                    gamma=float(sched_cfg.gamma),
                )
            elif sched_type == "multistep":
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=list(sched_cfg.milestones),
                    gamma=float(sched_cfg.gamma),
                )
            elif sched_type is None:
                self.lr_scheduler = None
            else:
                raise ValueError(f"Unsupported lr_scheduler type: {sched_type}")

    def _get_eval_model(self):
        """
        Use EMA model for validation / visualization if enabled,
        otherwise fall back to the raw training model.
        """
        if self.use_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_dab_loss = 0.0
        total_fake_dab_mean = 0.0
        total_real_dab_mean = 0.0
        total_fake_pos_frac = 0.0
        total_real_pos_frac = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.is_main_process(),
        )

        for batch in progress:
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.forward_step(batch)
                loss = outputs["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.forward_step(batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

            # EMA update after optimizer step
            if self.use_ema and self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            total_loss += outputs["loss"].item()
            total_diffusion_loss += outputs["diffusion_loss"].item()
            total_dab_loss += outputs["dab_loss"].item()
            total_fake_dab_mean += outputs["fake_dab_mean"].item()
            total_real_dab_mean += outputs["real_dab_mean"].item()
            total_fake_pos_frac += outputs["fake_pos_frac"].item()
            total_real_pos_frac += outputs["real_pos_frac"].item()

        denom = max(1, len(self.train_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "dab_loss": total_dab_loss / denom,
            "fake_dab_mean": total_fake_dab_mean / denom,
            "real_dab_mean": total_real_dab_mean / denom,
            "fake_pos_frac": total_fake_pos_frac / denom,
            "real_pos_frac": total_real_pos_frac / denom,
        }

    @torch.no_grad()
    def validate(self):
        eval_model = self._get_eval_model()
        eval_model.eval()

        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_dab_loss = 0.0
        total_fake_dab_mean = 0.0
        total_real_dab_mean = 0.0
        total_fake_pos_frac = 0.0
        total_real_pos_frac = 0.0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            disable=not self.is_main_process(),
        )

        for batch in progress:
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

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = eval_model(model_input, timesteps).sample
            else:
                pred = eval_model(model_input, timesteps).sample

            if self.noise_scheduler.prediction_type == "v_prediction":
                velocity = self.noise_scheduler.get_velocity(target, noise, timesteps)
                diffusion_loss = self.criterion(pred, velocity)
            else:
                diffusion_loss = self.criterion(pred, noise)

            if self.lambda_dab > 0.0:
                pred_x0 = self._predict_x0(
                    x_t=noisy_target.float(),
                    model_pred=pred.float(),
                    timesteps=timesteps,
                )
                dab_loss, dab_stats = self.dab_loss_fn(
                    fake_img=pred_x0.float(),
                    real_img=target.float(),
                )
            else:
                dab_loss = diffusion_loss.new_tensor(0.0)
                dab_stats = {
                    "fake_dab_mean": diffusion_loss.new_tensor(0.0),
                    "real_dab_mean": diffusion_loss.new_tensor(0.0),
                    "fake_pos_frac": diffusion_loss.new_tensor(0.0),
                    "real_pos_frac": diffusion_loss.new_tensor(0.0),
                }

            total = diffusion_loss.float() + self.lambda_dab * dab_loss.float()

            total_loss += total.item()
            total_diffusion_loss += diffusion_loss.item()
            total_dab_loss += dab_loss.item()
            total_fake_dab_mean += dab_stats["fake_dab_mean"].item()
            total_real_dab_mean += dab_stats["real_dab_mean"].item()
            total_fake_pos_frac += dab_stats["fake_pos_frac"].item()
            total_real_pos_frac += dab_stats["real_pos_frac"].item()

        denom = max(1, len(self.val_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "dab_loss": total_dab_loss / denom,
            "fake_dab_mean": total_fake_dab_mean / denom,
            "real_dab_mean": total_real_dab_mean / denom,
            "fake_pos_frac": total_fake_pos_frac / denom,
            "real_pos_frac": total_real_pos_frac / denom,
        }

    @torch.no_grad()
    def visualize_samples(self, epoch: int, num_samples: int = 4):
        eval_model = self._get_eval_model()
        eval_model.eval()

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
            noise_pred = eval_model(model_input, timesteps).sample
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
        import torchvision.utils as vutils
        vutils.save_image(grid, save_path, nrow=num_samples)

        print(f"[Visualization] Saved to {save_path}")
        return save_path

    def save_checkpoint(self, name: str, epoch: int):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.lr_scheduler is not None:
            ckpt["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.use_ema and self.ema_model is not None:
            ckpt["ema_model_state_dict"] = self.ema_model.state_dict()

        torch.save(ckpt, self.ckpt_dir / name)

    def train(self):
        try:
            for epoch in range(1, int(self.cfg.trainer.epochs) + 1):
                if self.cfg.trainer.distributed and hasattr(self.train_loader, "sampler"):
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics = self.train_one_epoch()
                val_metrics = self.validate()

                # Step LR scheduler once per epoch
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]

                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.4f} "
                    f"train_diff={train_metrics['diffusion_loss']:.4f} "
                    f"train_dab={train_metrics['dab_loss']:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_diff={val_metrics['diffusion_loss']:.4f} "
                    f"val_dab={val_metrics['dab_loss']:.4f}"
                )

                viz_path = None
                if self.is_main_process():
                    viz_path = self.visualize_samples(epoch)

                if self.cfg.trainer.checkpoint.save_last:
                    self.save_checkpoint("last.pt", epoch)

                if self.cfg.trainer.checkpoint.save_best and val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint("best.pt", epoch)

                current_lr = self.optimizer.param_groups[0]["lr"]

                self.wandb.log_metrics(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/diffusion_loss": train_metrics["diffusion_loss"],
                        "train/dab_loss": train_metrics["dab_loss"],
                        "train/fake_dab_mean": train_metrics["fake_dab_mean"],
                        "train/real_dab_mean": train_metrics["real_dab_mean"],
                        "train/fake_pos_frac": train_metrics["fake_pos_frac"],
                        "train/real_pos_frac": train_metrics["real_pos_frac"],
                        "val/loss": val_metrics["loss"],
                        "val/diffusion_loss": val_metrics["diffusion_loss"],
                        "val/dab_loss": val_metrics["dab_loss"],
                        "val/fake_dab_mean": val_metrics["fake_dab_mean"],
                        "val/real_dab_mean": val_metrics["real_dab_mean"],
                        "val/fake_pos_frac": val_metrics["fake_pos_frac"],
                        "val/real_pos_frac": val_metrics["real_pos_frac"],
                        "best/val_loss": self.best_val,
                        "lr": current_lr,
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
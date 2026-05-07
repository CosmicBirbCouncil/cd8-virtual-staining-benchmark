from __future__ import annotations

import torch
from tqdm import tqdm

from src.trainer.diffusion import DiffusionTrainer
from src.losses.hoptimus import HOptimusFeatureLoss


class DiffusionTrainerWithHOptimusLoss(DiffusionTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.lambda_feat = float(getattr(cfg.trainer, "lambda_feat", 0.0))

        self.feature_loss_fn = HOptimusFeatureLoss(
            model_name=str(getattr(cfg.trainer, "hoptimus_model_name", "hf-hub:bioptimus/H-optimus-1")),
            use_amp=bool(getattr(cfg.trainer, "hoptimus_use_amp", True)),
            amp_dtype=str(getattr(cfg.trainer, "hoptimus_amp_dtype", "float16")),
            resize_to=int(getattr(cfg.trainer, "hoptimus_resize_to", 224)),
        ).to(self.device)

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

        if self.lambda_feat > 0.0:
            pred_x0 = self._predict_x0(
                x_t=noisy_target.float(),
                model_pred=pred.float(),
                timesteps=timesteps,
            )

            feat_loss, feat_stats = self.feature_loss_fn(
                fake_img=pred_x0.float(),
                real_img=target.float(),
            )
        else:
            feat_loss = diffusion_loss.new_tensor(0.0)
            feat_stats = {
                "fake_feat_norm": diffusion_loss.new_tensor(0.0),
                "real_feat_norm": diffusion_loss.new_tensor(0.0),
            }

        total_loss = diffusion_loss.float() + self.lambda_feat * feat_loss.float()

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss.detach(),
            "feat_loss": feat_loss.detach(),
            **feat_stats,
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_feat_loss = 0.0
        total_fake_feat_norm = 0.0
        total_real_feat_norm = 0.0

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
            total_feat_loss += outputs["feat_loss"].item()
            total_fake_feat_norm += outputs["fake_feat_norm"].item()
            total_real_feat_norm += outputs["real_feat_norm"].item()

        denom = max(1, len(self.train_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "feat_loss": total_feat_loss / denom,
            "fake_feat_norm": total_fake_feat_norm / denom,
            "real_feat_norm": total_real_feat_norm / denom,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_feat_loss = 0.0
        total_fake_feat_norm = 0.0
        total_real_feat_norm = 0.0

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
            total_feat_loss += outputs["feat_loss"].item()
            total_fake_feat_norm += outputs["fake_feat_norm"].item()
            total_real_feat_norm += outputs["real_feat_norm"].item()

        denom = max(1, len(self.val_loader))
        return {
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion_loss / denom,
            "feat_loss": total_feat_loss / denom,
            "fake_feat_norm": total_fake_feat_norm / denom,
            "real_feat_norm": total_real_feat_norm / denom,
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
                    f"train_feat={train_metrics['feat_loss']:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_diff={val_metrics['diffusion_loss']:.4f} "
                    f"val_feat={val_metrics['feat_loss']:.4f}"
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
                        "train/feat_loss": train_metrics["feat_loss"],
                        "train/fake_feat_norm": train_metrics["fake_feat_norm"],
                        "train/real_feat_norm": train_metrics["real_feat_norm"],
                        "val/loss": val_metrics["loss"],
                        "val/diffusion_loss": val_metrics["diffusion_loss"],
                        "val/feat_loss": val_metrics["feat_loss"],
                        "val/fake_feat_norm": val_metrics["fake_feat_norm"],
                        "val/real_feat_norm": val_metrics["real_feat_norm"],
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
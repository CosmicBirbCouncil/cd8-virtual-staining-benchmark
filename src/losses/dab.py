from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DABLoss(nn.Module):
    """
    Batch-level DAB calibration loss for weakly aligned / unpaired training.

    Matches:
    - mean DAB intensity
    - mean DAB-positive fraction
    - optional DAB histogram

    This is not a pixelwise paired loss.
    """

    def __init__(
        self,
        pos_threshold: float = 0.05,
        sharpness: float = 40.0,
        use_hist: bool = True,
        hist_bins: int = 16,
    ):
        super().__init__()
        self.pos_threshold = pos_threshold
        self.sharpness = sharpness
        self.use_hist = use_hist
        self.hist_bins = hist_bins

    def normalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1, 1) + 1.0) / 2.0

    def dab_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable proxy for DAB-like brown staining.
        Input: [B,3,H,W] in [-1,1] or [0,1]
        Output: [B,1,H,W]
        """
        x = self.normalize_to_01(x)

        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        brownness = torch.relu(0.5 * (r + g) - b)
        darkness = 1.0 - x.mean(dim=1, keepdim=True)

        return brownness * darkness

    def soft_positive_fraction(self, dab_map: torch.Tensor) -> torch.Tensor:
        """
        Smooth approximation of fraction of DAB-positive pixels per image.
        Returns [B]
        """
        soft_mask = torch.sigmoid(self.sharpness * (dab_map - self.pos_threshold))
        return soft_mask.mean(dim=(1, 2, 3))

    def soft_histogram(self, values: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft histogram over values in [0,1].
        values: [N]
        returns: [hist_bins]
        """
        values = values.clamp(0.0, 1.0)
        centers = torch.linspace(
            0.0,
            1.0,
            self.hist_bins,
            device=values.device,
            dtype=values.dtype,
        )
        sigma = 1.0 / self.hist_bins
        weights = torch.exp(-0.5 * ((values[:, None] - centers[None, :]) / sigma) ** 2)
        hist = weights.sum(dim=0)
        hist = hist / (hist.sum() + 1e-8)
        return hist

    def forward(
        self,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        fake_img: generated / pred_x0 image [B,3,H,W]
        real_img: target image [B,3,H,W]
        """
        fake_dab = self.dab_proxy(fake_img)
        real_dab = self.dab_proxy(real_img)

        fake_mean = fake_dab.mean(dim=(1, 2, 3))
        real_mean = real_dab.mean(dim=(1, 2, 3))
        loss_mean = F.mse_loss(fake_mean.mean(), real_mean.mean())

        fake_pos_frac = self.soft_positive_fraction(fake_dab)
        real_pos_frac = self.soft_positive_fraction(real_dab)
        loss_pos = F.mse_loss(fake_pos_frac.mean(), real_pos_frac.mean())

        if self.use_hist:
            fake_hist = self.soft_histogram(fake_dab.flatten())
            real_hist = self.soft_histogram(real_dab.flatten())
            loss_hist = F.mse_loss(fake_hist, real_hist)
        else:
            loss_hist = fake_dab.new_tensor(0.0)

        total = loss_mean + loss_pos + loss_hist

        stats = {
            "fake_dab_mean": fake_mean.mean().detach(),
            "real_dab_mean": real_mean.mean().detach(),
            "fake_pos_frac": fake_pos_frac.mean().detach(),
            "real_pos_frac": real_pos_frac.mean().detach(),
            "dab_loss_mean": loss_mean.detach(),
            "dab_loss_pos": loss_pos.detach(),
            "dab_loss_hist": loss_hist.detach(),
        }

        return total, stats
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HOptimusFeatureLoss(nn.Module):
    """
    Feature-space L1 loss using frozen H-Optimus-1 embeddings.

    Compares:
        synthetic CD8 (pred_x0) vs real CD8

    This is NOT a registration loss.
    It encourages synthetic tiles to look more like real CD8 in feature space.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:bioptimus/H-optimus-1",
        use_amp: bool = True,
        amp_dtype: str = "float16",
        resize_to: int = 224,
    ):
        super().__init__()

        self.resize_to = resize_to
        self.use_amp = use_amp
        self.amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16

        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1)
        std = torch.tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _normalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1] -> [0,1]
        return (x.clamp(-1, 1) + 1.0) / 2.0

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_to_01(x)
        x = F.interpolate(
            x,
            size=(self.resize_to, self.resize_to),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)

        # Keep gradients w.r.t. input image, but freeze encoder weights.
        if x.device.type == "cuda" and self.use_amp:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)

        return feats

    def forward(
        self,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fake_feats = self.extract_features(fake_img)
        real_feats = self.extract_features(real_img)

        loss = F.l1_loss(fake_feats, real_feats)

        stats = {
            "fake_feat_norm": fake_feats.norm(dim=1).mean().detach(),
            "real_feat_norm": real_feats.norm(dim=1).mean().detach(),
        }

        return loss, stats
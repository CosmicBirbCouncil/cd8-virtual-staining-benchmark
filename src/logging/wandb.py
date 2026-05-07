from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf


class WandbLogger:
    def __init__(self, cfg, enabled: bool, is_main_process: bool):
        self.cfg = cfg
        self.enabled = enabled and is_main_process
        self.run = None

    def init(self) -> None:
        if not self.enabled:
            return

        config_dict = OmegaConf.to_container(self.cfg, resolve=True)

        self.run = wandb.init(
            project=self.cfg.logging.wandb_project,
            entity=self.cfg.logging.wandb_entity,
            name=self.cfg.logging.wandb_run_name,
            tags=list(getattr(self.cfg.logging, "wandb_tags", [])),
            config=config_dict,
        )

        if getattr(self.cfg.logging, "wandb_watch_model", False):
            model = getattr(self, "_model_to_watch", None)
            if model is not None:
                self.run.watch(
                    model,
                    log="gradients",
                    log_freq=1000,
                    log_graph=False,
                )

    def set_model(self, model) -> None:
        self._model_to_watch = model

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics, step=step)

    def log_image(self, key: str, image_path: str | os.PathLike, step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return

        image_path = Path(image_path)
        if not image_path.exists():
            return

        self.run.log({key: wandb.Image(str(image_path))}, step=step)

    def log_images(self, images: dict[str, str | os.PathLike], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return

        payload = {}
        for key, image_path in images.items():
            image_path = Path(image_path)
            if image_path.exists():
                payload[key] = wandb.Image(str(image_path))

        if payload:
            self.run.log(payload, step=step)

    def log_artifact_file(self, path: str | os.PathLike, name: str, artifact_type: str = "file") -> None:
        if not self.enabled or self.run is None:
            return

        path = Path(path)
        if not path.exists():
            return

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.run = None
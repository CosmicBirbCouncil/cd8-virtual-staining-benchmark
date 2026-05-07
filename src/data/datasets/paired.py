from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PairedDiffusionDataset(Dataset):
    REQUIRED_COLUMNS = {
        "case_id",
        "input_tile",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Tile manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self._validate_columns(self.df)
        # self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in manifest: {self.manifest_path}"
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    def _load_tile(self, input_tile_path: Path, target_tile_path: Path) -> np.ndarray:
        if not input_tile_path.exists():
            raise FileNotFoundError(f"Missing input tile: {input_tile_path}")
        if not target_tile_path.exists():
            raise FileNotFoundError(f"Missing target tile: {target_tile_path}")

        if input_tile_path.suffix == ".npy" and target_tile_path.suffix == ".npy":
            return np.load(input_tile_path)[:, :, :3], np.load(target_tile_path)[:, :, :3]
        else:
            raise Exception('Current implementation only allows for .npy files.')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        input_tile_path = Path(row["input_tile"])
        target_tile_path = Path(row["target_tile"])
        
        condition_image, target_image = self._load_tile(input_tile_path, target_tile_path)

        if self.transform is not None:
            transformed = self.transform(
                image=condition_image,
                target_image=target_image,
            )
            condition_image = transformed["image"]
            target_image = transformed["target_image"]

        sample = {
            "condition_image": condition_image,
            "target_image": target_image,
            "case_id": row["case_id"],
        }

        if "tile_id" in row.index:
            sample["tile_id"] = row["tile_id"]
        if "x" in row.index:
            sample["x"] = row["x"]
        if "y" in row.index:
            sample["y"] = row["y"]
        if "level" in row.index:
            sample["level"] = row["level"]

        return sample
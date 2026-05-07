from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


def load_tile(path: Union[str, Path]) -> np.ndarray:
    arr = np.load(Path(path))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def visualize_tile_pairings_side_by_side(
    manifest_csv: str,
    output_path: str = "debug/tile_pairings_side_by_side.png",
    n_samples: int = 16,
    seed: int = 42,
    input_col: str = "input_tile",
    target_col: str = "target_tile",
    generated_col: str = "generated_tile",
):
    df = pd.read_csv(manifest_csv)

    required_cols = [generated_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows found after dropping missing tile paths.")

    sample_rows = df.sample(
        n=min(n_samples, len(df)),
        random_state=seed
    ).reset_index(drop=True)

    n = len(sample_rows)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(9, n * 3),
        squeeze=False,
    )

    for i, row in sample_rows.iterrows():
        real_path = f"/ix/rbao/shared/rbao_qug14_ngl18/cd8_patches/patch_512_40x/"
        file_name = Path(row[generated_col]).name
        he = Path (real_path) / row["case_id"] / "HE" / file_name
        ihc = Path (real_path) / row["case_id"] / "CD8" / file_name
        he = load_tile(he)
        real = load_tile(ihc)
        fake = load_tile(row[generated_col])

        # HE
        axes[i, 0].imshow(he)
        axes[i, 0].set_title("HE", fontsize=10)
        axes[i, 0].axis("off")

        # Real CD8
        axes[i, 1].imshow(real)
        axes[i, 1].set_title("Real CD8", fontsize=10)
        axes[i, 1].axis("off")

        # Generated CD8
        axes[i, 2].imshow(fake)
        axes[i, 2].set_title("Generated CD8", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    visualize_tile_pairings_side_by_side(
        manifest_csv="checkpoints_cd8_pos/eval/dab_eval_results.csv",
        output_path="debug/tile_pairings_side_by_side.png",
        n_samples=16,
        seed=42,
    )
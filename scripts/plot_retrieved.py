from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_npy_rgb(path: str | Path) -> np.ndarray:
    arr = np.load(path)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected array shape for {path}: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)

    if arr.max() > 1.5:
        arr = arr / 255.0
    elif arr.min() < 0.0:
        arr = (arr + 1.0) / 2.0

    return np.clip(arr, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--he_col", type=str, default="input_tile")
    parser.add_argument("--cd8_col", type=str, default="target_tile")
    parser.add_argument("--retrieved_col", type=str, default="retrieved_cd8")
    parser.add_argument("--score_col", type=str, default="retrieval_score")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for col in [args.he_col, args.cd8_col, args.retrieved_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ✅ Random sample of 4
    n_samples = min(4, len(df))
    df_sample = df.sample(n=n_samples).reset_index(drop=True)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, row in df_sample.iterrows():
        he = load_npy_rgb(row[args.he_col])
        cd8 = load_npy_rgb(row[args.cd8_col])
        retrieved = load_npy_rgb(row[args.retrieved_col])

        # Titles
        score = None
        if args.score_col in df.columns and pd.notna(row[args.score_col]):
            score = float(row[args.score_col])

        # H&E
        axes[i, 0].imshow(he)
        axes[i, 0].set_title("H&E")
        axes[i, 0].axis("off")

        # CD8
        axes[i, 1].imshow(cd8)
        axes[i, 1].set_title("CD8")
        axes[i, 1].axis("off")

        # Retrieved
        title = "Retrieved CD8"
        if score is not None:
            title += f"\nscore={score:.4f}"

        axes[i, 2].imshow(retrieved)
        axes[i, 2].set_title(title)
        axes[i, 2].axis("off")

    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
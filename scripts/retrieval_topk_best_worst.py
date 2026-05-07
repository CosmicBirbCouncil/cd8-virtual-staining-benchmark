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


def plot_rows(
    df: pd.DataFrame,
    he_col: str,
    raw_cd8_col: str,
    retrieved_col: str,
    score_col: str,
    save_path: str | Path,
    title: str,
) -> None:
    n = len(df)
    if n == 0:
        print(f"[WARN] No rows to plot for {save_path}")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (_, row) in enumerate(df.iterrows()):
        he = load_npy_rgb(row[he_col])
        raw_cd8 = load_npy_rgb(row[raw_cd8_col])
        retrieved = load_npy_rgb(row[retrieved_col])

        score = float(row[score_col]) if pd.notna(row[score_col]) else None

        row_title_parts = []
        if "case_id" in row:
            row_title_parts.append(f"case={row['case_id']}")
        if "tile_id" in row:
            row_title_parts.append(f"tile={row['tile_id']}")
        if score is not None:
            row_title_parts.append(f"score={score:.4f}")

        row_title = " | ".join(row_title_parts)

        axes[i, 0].imshow(he)
        axes[i, 0].set_title(f"H&E\n{row_title}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(raw_cd8)
        axes[i, 1].set_title("Raw CD8", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(retrieved)
        axes[i, 2].set_title("Retrieved CD8", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--he_col", type=str, default="input_tile")
    parser.add_argument("--raw_cd8_col", type=str, default="target_tile")
    parser.add_argument("--retrieved_col", type=str, default="retrieved_cd8")
    parser.add_argument("--score_col", type=str, default="retrieval_score")

    parser.add_argument("--k", type=int, default=8)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = [args.he_col, args.raw_cd8_col, args.retrieved_col, args.score_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    k = min(args.k, len(df))

    df_sorted = df.sort_values(args.score_col, ascending=False).reset_index(drop=True)
    topk_df = df_sorted.head(k)
    worstk_df = df_sorted.tail(k).sort_values(args.score_col, ascending=True).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_rows(
        df=topk_df,
        he_col=args.he_col,
        raw_cd8_col=args.raw_cd8_col,
        retrieved_col=args.retrieved_col,
        score_col=args.score_col,
        save_path=output_dir / f"top_{k}.png",
        title=f"Top-{k} Retrieval Matches",
    )

    plot_rows(
        df=worstk_df,
        he_col=args.he_col,
        raw_cd8_col=args.raw_cd8_col,
        retrieved_col=args.retrieved_col,
        score_col=args.score_col,
        save_path=output_dir / f"worst_{k}.png",
        title=f"Worst-{k} Retrieval Matches",
    )


if __name__ == "__main__":
    main()
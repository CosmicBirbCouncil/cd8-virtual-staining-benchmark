from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import random
from tqdm import tqdm


def load_npy_as_image(path: Path):
    arr = np.load(path)

    # Take first 3 channels if needed
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    # Normalize to [0,255] if needed
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    return Image.fromarray(arr)


def sample_he_tiles_from_manifest(
    manifest_path: str,
    output_dir: str,
    num_samples_per_slide: int = 64,
    seed: int = 42,
):
    random.seed(seed)

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    if "case_id" not in df.columns or "input_tile" not in df.columns:
        raise ValueError("Manifest must contain 'case_id' and 'input_tile' columns")

    grouped = df.groupby("case_id")

    for case_id, group in tqdm(grouped, desc="Processing WSIs"):
        case_dir = output_dir / str(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = group["input_tile"].tolist()

        # Sample tiles
        sampled_paths = random.sample(
            tile_paths,
            min(num_samples_per_slide, len(tile_paths))
        )

        for i, tile_path in enumerate(sampled_paths):
            tile_path = Path(tile_path)

            try:
                img = load_npy_as_image(tile_path)

                save_path = case_dir / f"{i:03d}.png"
                img.save(save_path)

            except Exception as e:
                print(f"[ERROR] {tile_path}: {e}")


if __name__ == "__main__":
    sample_he_tiles_from_manifest(
        manifest_path="/ix/rbao/shared/rbao_qug14_ngl18/cd8_manifests/patch_512/cd8_train_manifest.csv",
        output_dir="he_samples_train",
        num_samples_per_slide=64,
    )
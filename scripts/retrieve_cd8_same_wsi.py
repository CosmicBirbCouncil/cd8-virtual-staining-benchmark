from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


import numpy as np
import torch.nn.functional as F


class HOptimusEmbedder:
    def __init__(
        self,
        model_name: str = "hf-hub:bioptimus/H-optimus-1",
        device: str = "cuda",
        batch_size: int = 64,
        amp: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.amp = amp and self.device.type == "cuda"

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        
        self.model.to(self.device)
        self.model.eval()

        # normalization only (resize handled manually)
        self.mean = torch.tensor([0.707223, 0.578729, 0.703617]).view(3, 1, 1)
        self.std = torch.tensor([0.211883, 0.230117, 0.177517]).view(3, 1, 1)

    def load_image(self, path: str | Path) -> torch.Tensor:
        arr = np.load(path)

        # Ensure shape (H, W, 3)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)

        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got {arr.shape}")

        # Convert to float tensor
        tensor = torch.from_numpy(arr).float()

        # If uint8 → scale
        if tensor.max() > 1.5:
            tensor = tensor / 255.0

        # HWC → CHW
        tensor = tensor.permute(2, 0, 1)

        # Resize to 224x224 (H-Optimus requirement)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalize
        tensor = (tensor - self.mean) / self.std

        return tensor
    
    @torch.no_grad()
    def embed_paths(self, paths: List[str]) -> torch.Tensor:
        feats = []

        for start in tqdm(range(0, len(paths), self.batch_size), leave=False):
            batch_paths = paths[start:start + self.batch_size]
            batch = torch.stack([self.load_image(p) for p in batch_paths], dim=0).to(self.device)

            if self.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_feats = self.model(batch)
            else:
                batch_feats = self.model(batch)

            batch_feats = F.normalize(batch_feats.float(), dim=1)
            feats.append(batch_feats.cpu())

        return torch.cat(feats, dim=0)


def batched_cosine_top1(
    query_feats: torch.Tensor,
    key_feats: torch.Tensor,
    key_paths: List[str],
    batch_size: int = 512,
) -> tuple[List[str], List[float], List[int]]:
    """
    query_feats: [Nq, D] normalized
    key_feats:   [Nk, D] normalized
    Returns top-1 retrieved path, score, and index for each query.
    """
    retrieved_paths: List[str] = []
    retrieved_scores: List[float] = []
    retrieved_indices: List[int] = []

    key_feats = key_feats.float()

    for start in range(0, query_feats.shape[0], batch_size):
        q = query_feats[start:start + batch_size].float()  # [B, D]
        sims = q @ key_feats.T  # cosine because normalized
        top_scores, top_idx = sims.max(dim=1)

        retrieved_paths.extend([key_paths[i] for i in top_idx.tolist()])
        retrieved_scores.extend(top_scores.tolist())
        retrieved_indices.extend(top_idx.tolist())

    return retrieved_paths, retrieved_scores, retrieved_indices


def retrieve_same_group(
    df: pd.DataFrame,
    group_col: str,
    he_col: str,
    cd8_col: str,
    embedder: HOptimusEmbedder,
) -> pd.DataFrame:
    output_rows = []

    grouped = df.groupby(group_col)

    for group_value, group_df in tqdm(grouped, total=df[group_col].nunique(), desc="Groups"):
        group_df = group_df.reset_index(drop=True)

        he_paths = group_df[he_col].astype(str).tolist()
        cd8_paths = group_df[cd8_col].astype(str).tolist()

        # Unique candidate pools within the same group
        unique_he_paths = list(dict.fromkeys(he_paths))
        unique_cd8_paths = list(dict.fromkeys(cd8_paths))

        if len(unique_he_paths) == 0 or len(unique_cd8_paths) == 0:
            continue

        he_feat_map: Dict[str, torch.Tensor] = {}
        cd8_feat_map: Dict[str, torch.Tensor] = {}

        he_feats = embedder.embed_paths(unique_he_paths)
        cd8_feats = embedder.embed_paths(unique_cd8_paths)

        for p, f in zip(unique_he_paths, he_feats):
            he_feat_map[p] = f
        for p, f in zip(unique_cd8_paths, cd8_feats):
            cd8_feat_map[p] = f

        query_feats = torch.stack([he_feat_map[p] for p in he_paths], dim=0)
        key_feats = torch.stack([cd8_feat_map[p] for p in unique_cd8_paths], dim=0)

        retrieved_paths, retrieved_scores, retrieved_indices = batched_cosine_top1(
            query_feats=query_feats,
            key_feats=key_feats,
            key_paths=unique_cd8_paths,
            batch_size=512,
        )

        for i, row in group_df.iterrows():
            out = dict(row)
            out["retrieved_cd8"] = retrieved_paths[i]
            out["retrieval_score"] = float(retrieved_scores[i])
            out["retrieval_group"] = group_value
            output_rows.append(out)

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--group_col", type=str, default="case_id")
    parser.add_argument("--he_col", type=str, default="input_tile")
    parser.add_argument("--cd8_col", type=str, default="target_tile")

    parser.add_argument("--model_name", type=str, default="hf-hub:bioptimus/H-optimus-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    for col in [args.group_col, args.he_col, args.cd8_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.input_csv}")

    embedder = HOptimusEmbedder(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        amp=args.amp,
    )

    out_df = retrieve_same_group(
        df=df,
        group_col=args.group_col,
        he_col=args.he_col,
        cd8_col=args.cd8_col,
        embedder=embedder,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved retrieval results to: {output_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()
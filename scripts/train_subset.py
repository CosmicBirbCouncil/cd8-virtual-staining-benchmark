from pathlib import Path
import pandas as pd
import numpy as np
import argparse


def sample_per_case(
    df: pd.DataFrame,
    case_col: str = "case_id",
    max_per_case: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    sampled_dfs = []

    for case_id, group in df.groupby(case_col):
        if len(group) <= max_per_case:
            sampled = group
        else:
            sampled = group.sample(n=max_per_case, random_state=seed)

        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_per_case", type=int, default=250)
    parser.add_argument("--case_col", type=str, default="case_id")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"Loading manifest: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Total rows: {len(df)}")
    print(f"Unique cases: {df[args.case_col].nunique()}")

    sampled_df = sample_per_case(
        df,
        case_col=args.case_col,
        max_per_case=args.max_per_case,
        seed=args.seed,
    )

    print(f"Sampled rows: {len(sampled_df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_csv, index=False)

    print(f"Saved sampled manifest to: {output_csv}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_tiles(case_dir: Path, pattern: str = "*.npy", recursive: bool = True) -> list[Path]:
    if recursive:
        files = sorted(case_dir.rglob(pattern))
    else:
        files = sorted(case_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def infer_slide_id(case_dir: Path, tile_path: Path) -> str | None:
    rel_parts = tile_path.relative_to(case_dir).parts
    if len(rel_parts) >= 2:
        return rel_parts[0]
    return None


def build_case_tile_manifest(
    case_dir: Path,
    output_csv: Path,
    pattern: str = "*.npy",
    recursive: bool = True,
    path_col: str = "input_tile",
) -> Path:
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    if not case_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {case_dir}")

    case_id = case_dir.name
    tile_paths = find_tiles(case_dir=case_dir, pattern=pattern, recursive=recursive)

    if len(tile_paths) == 0:
        raise ValueError(f"No tile files found in {case_dir} with pattern '{pattern}'")

    rows = []
    for tile_id, tile_path in enumerate(tile_paths):
        row = {
            "case_id": case_id,
            "tile_id": tile_id,
            path_col: str(tile_path.resolve()),
            "filename": tile_path.name,
            "relative_path": str(tile_path.relative_to(case_dir)),
        }

        slide_id = infer_slide_id(case_dir, tile_path)
        if slide_id is not None:
            row["slide_id"] = slide_id

        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a case directory and generate a CSV listing all tile files."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to the case directory to scan.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <case_dir>/<case_id>_tiles.csv",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern for tile files.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Scan only the top level of the case directory.",
    )
    parser.add_argument(
        "--path-col",
        type=str,
        default="input_tile",
        help="Name of the CSV column containing tile paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else case_dir / f"{case_dir.name}_tiles.csv"
    )

    out_path = build_case_tile_manifest(
        case_dir=case_dir,
        output_csv=output_csv,
        pattern=args.pattern,
        recursive=not args.non_recursive,
        path_col=args.path_col,
    )

    print(f"Saved case tile manifest to: {out_path}")


if __name__ == "__main__":
    main()
#!/bin/bash

set -e

PATCH_DIR="/ix/rbao/shared/rbao_qug14_ngl18/cd8_patches/patch_512"
OUT_DIR="/ix/rbao/shared/rbao_qug14_ngl18/cd8_manifests/patch_512"

mkdir -p "$OUT_DIR"

for i in $(seq 1 82); do
  case_id=$(printf "%03d" "$i")

  case_dir="${PATCH_DIR}/T${case_id}"
  output_csv="${OUT_DIR}/T${case_id}_tiles.csv"

  if [ ! -d "$case_dir" ]; then
    echo "Skipping missing case: $case_dir"
    continue
  fi

  echo "Processing case T${case_id}..."

  python build_tile_manifest.py \
    --case-dir "$case_dir" \
    --output-csv "$output_csv"

done

echo "Done."
#!/bin/bash
#SBATCH --job-name=hop_embeddings
#SBATCH --output=logs/hop_embeddings_%j.out
#SBATCH --error=logs/hop_embeddings_%j.err
#SBATCH --cluster=cpace
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

export HUGGINGFACE_HUB_CACHE=/ihome/atafti/ngl18/.cache/huggingface/hub/

set -euo pipefail

mkdir -p logs

module load python/ondemand-jupyter-python3.11
source activate /ix3/cpace/ngl18/envs/rapids-25.08/ 

python scripts/retrieve_cd8_same_wsi.py --input_csv /ix/rbao/shared/rbao_qug14_ngl18/cd8_manifests/patches_512_40x_cd8_pos_split_test.csv --output_csv /ix/rbao/shared/rbao_qug14_ngl18/cd8_manifests/patches_512_40x_cd8_pos_split_test_hop_retrieval.csv
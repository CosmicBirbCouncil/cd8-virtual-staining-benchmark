#!/bin/bash
#SBATCH --job-name=vs_diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Go to project root
cd /ix3/cpace/ngl18/projects/virtual-staining-v2

# Make logs dir if needed
mkdir -p logs

# Activate environment
module load python/ondemand-jupyter-python3.11
source activate /ix3/cpace/ngl18/envs/rapids-25.08/ 

# Make project importable
export PYTHONPATH=.

# Keep thread usage under control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: better error traces
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# NCCL / networking
export NCCL_DEBUG=INFO
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501

export WANDB_API_KEY="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"
echo "Running on host: $MASTER_ADDR"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

torchrun \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m src.cli.run_diffusion
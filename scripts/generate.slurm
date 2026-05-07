#!/bin/bash
#SBATCH --job-name=vs_diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Go to project root
cd /ix3/cpace/ngl18/projects/virtual-staining

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

echo "Running on host: $MASTER_ADDR"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

torchrun \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m src.cli.generate
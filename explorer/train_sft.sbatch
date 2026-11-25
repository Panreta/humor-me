#!/bin/bash
# Step 4: SFT (--stage sft) or O-GRPO (--stage ogrpo --init_adapter <sft>/final) on runs/sets.
# Submit from the repo root: sbatch explorer/train_sft.sbatch [extra flags]. Resumable: resubmit the same
# command after a time limit or node failure and it continues where it stopped.
#SBATCH --job-name=train_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m explorer.train_sft "$@"

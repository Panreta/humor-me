#!/bin/bash
# Step 1a: LLM-written premises with the 32B teacher (option A; set HF_HUB_OFFLINE=0 once to download it).
# Submit from the repo root: sbatch explorer/gen_premises.sbatch [extra flags]. Resumable: resubmit the same
# command after a time limit or node failure and it continues where it stopped.
#SBATCH --job-name=gen_premises
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m explorer.gen_premises "$@"

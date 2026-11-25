#!/bin/bash
# Step 5: base vs SFT vs DPO vs O-GRPO on the held-out premises. Two subcommands, run in order:
#   sbatch explorer/eval_generators.sbatch generate --models base sft=<dir> dpo=<dir> ogrpo=<dir>
#   sbatch explorer/eval_generators.sbatch judge
# Submit from the repo root. Resumable: generate skips models already in runs/eval/lines.jsonl.
#SBATCH --job-name=eval_generators
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
python -m explorer.eval_generators "$@"

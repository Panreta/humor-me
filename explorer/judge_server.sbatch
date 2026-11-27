#!/bin/bash
# Serve the DPO judge for a show: sbatch explorer/judge_server.sbatch [--dtype fp16] [extra flags]
# Then on the laptop: ssh -L 8765:$(cat runs/judge_server.addr | cut -d: -f1):8765 <user>@login.explorer.northeastern.edu
#SBATCH --job-name=judge_server
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export HF_HUB_OFFLINE=1
python -m explorer.judge_server --adapter "$COURSE_DIR/runs/dpo_judge_v1/final" --dtype fp16 "$@"

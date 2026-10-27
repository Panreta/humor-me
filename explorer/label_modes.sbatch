#!/bin/bash
# Batch version of label_modes.py: survives closing the browser, and isn't killed as an idle
# OOD session. Submit from the repo root: sbatch explorer/label_modes.sbatch
#SBATCH --job-name=humor-label
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n "${N:-30000}" ${MODEL:+--model "$MODEL"}

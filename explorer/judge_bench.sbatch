#!/bin/bash
# Start the judge server, measure it (explorer/judge_bench.py), stop. sbatch explorer/judge_bench.sbatch
#SBATCH --job-name=judge_bench
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m explorer.judge_server --adapter "$COURSE_DIR/runs/dpo_judge_v1/final" --dtype fp16 --addr-file runs/judge_bench.addr &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf --noproxy "*" localhost:8765/health >/dev/null && break; sleep 5; done
python -m explorer.judge_bench --url http://localhost:8765 --out runs/judge_bench.json

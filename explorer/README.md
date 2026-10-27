# Explorer (Northeastern HPC) jobs

GPU batch work only. Nothing on the laptop may depend on these jobs having run: the sim and
bandits start on placeholder priors.

## One-time setup

1. OOD dashboard → Courses → JupyterLab (ml-courses). Partition: the GPU one (`courses-gpu`),
   1 GPU, 8 CPUs, 32 GB RAM, 2 h. Note which GPU you get (`nvidia-smi`): A100/H100/L40 run
   a 7B model easily; on a 16 GB card (T4/P100/V100-16) use `--model Qwen/Qwen2.5-3B-Instruct`.
2. In a JupyterLab terminal:
   ```bash
   cd /scratch/$USER                      # home quota is small; models are ~15 GB
   git clone <repo url> humor-me && cd humor-me
   python -m venv .venv && source .venv/bin/activate
   pip install vllm                       # brings a matching torch
   export HF_HOME=/scratch/$USER/hf       # keep model downloads off home
   ```
   If `python` is too old for vLLM, load a newer one first (`module avail python`).
3. Copy `data/processed/rjokes/train.jsonl` up (OOD Files → Upload, or `scp`) into the same
   path. It's gitignored on purpose: joke text never goes into git.

## Jobs

| Job | Command | Output |
|---|---|---|
| Smoke test | `python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 50 --out /tmp/smoke.jsonl` | Throughput + parse failures |
| Mode + safety labels | `python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 30000` | `data/processed/rjokes/modes.jsonl` (ids + labels, no text) |

After the run, copy `modes.jsonl` back to the laptop, then spot-check 100 labels by hand
before the sim uses them (record agreement in dev-log.md).

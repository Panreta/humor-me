# Explorer HPC

**Where it fits:** offline GPU work that the laptop (RTX 4060) and Gemini's free tier can't
do in time. Nothing in the live loop or the sim depends on it.
**Used in:** `explorer/` (labeling job, batch script, env), later the joke scorer.

## Concepts

| Term | Meaning here |
|---|---|
| **Slurm** | The scheduler: you ask for resources, it queues and runs your job |
| **Partition** | A pool of nodes with limits. Ours: `courses` (CPU) and `courses-gpu` |
| **`sbatch`** | Submit a script to run unattended; survives closing the browser |
| **`srun` / OOD JupyterLab** | Interactive sessions; good for smoke tests, wasteful for long runs |
| **OOD** | Open OnDemand, the web portal (ood.explorer.northeastern.edu) |
| **vLLM** | Batched LLM inference server/library; why 30k labels take minutes, not days |

## Limits that shaped our design (CS5180 course access)

- `courses-gpu`: 8 h max, **1 running job**, **12 GB RAM**; idle ~1 h → killed.
  → long runs use `sbatch`; the script loads only the sample, not 400k jokes.
- Student dir `/courses/CS5180.<term>/students/<user>` is **deleted after grades**.
  → copy results back after every run; outputs hold ids and labels, never joke text.

## Further reading

- [NURC course guide](https://rc-docs.northeastern.edu/en/explorer-main/classroom/class_guide.html):
  partitions, storage, OOD.
- [Slurm quick start](https://slurm.schedmd.com/quickstart.html): `sbatch`, `squeue`,
  `scancel`.
- [vLLM docs](https://docs.vllm.ai/): offline batch inference (`LLM.chat`).

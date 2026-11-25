"""Step 0 data: rJokes upvote pairs for training and testing the A/B judge (docs/plans/sft-dpo-ogrpo.md).

Usage (from the repo root, on the laptop or the cluster):
    python explorer/build_judge_pairs.py --splits data/processed/rjokes --out /scratch/$USER/dpo

Reads {train,dev,test}.jsonl from `python -m pipeline.rjokes` and writes dpo_{train,dev,test}.jsonl with
records {"prompt", "chosen", "rejected"}, the format explorer/train_dpo.py and explorer/eval_judge.py read.
Rule (fixed before any judge sees the pairs, DPO_training.md):
1. Keep jokes with a known year and no keyword flag.
2. Cell = year x length quartile (cuts from the split's own word counts), so a pair compares jokes of similar
   length from the same year: both move upvotes.
3. In each cell, pair a high joke (pct >= 0.8) with a low one (pct <= 0.3), each joke used once.
4. Write every pair in both orders, so "A" and "B" are each the right answer half the time.
5. Shuffle with seed 17.
The outputs hold Reddit text: keep them off git (decisions.md 2026-10-31).
This is the laptop's make_dpo_pairs.py (2026-11-14) made importable; same seed, same files.
"""
import argparse
import json
import pathlib
import random

import numpy as np

PROMPT = "Which joke is funnier?\nJoke A: {a}\nJoke B: {b}\nAnswer with A or B."
HIGH, LOW = 0.8, 0.3


def build(rows: list[dict], seed: int = 17) -> list[dict]:
    """Judge records for one split, in shuffled order."""
    rows = [r for r in rows if r.get("year") is not None and not r.get("flags")]
    lengths = [len(r["text"].split()) for r in rows]
    cuts = np.quantile(lengths, [0.25, 0.5, 0.75])
    cells = {}
    for r, n in zip(rows, lengths):
        cells.setdefault((r["year"], int(np.searchsorted(cuts, n))), []).append(r)

    rng = random.Random(seed)
    out = []
    for group in cells.values():
        high = [r for r in group if r["pct"] >= HIGH]
        low = [r for r in group if r["pct"] <= LOW]
        rng.shuffle(high)
        rng.shuffle(low)
        for h, l in zip(high, low):
            out.append({"prompt": PROMPT.format(a=h["text"], b=l["text"]), "chosen": "A", "rejected": "B"})
            out.append({"prompt": PROMPT.format(a=l["text"], b=h["text"]), "chosen": "B", "rejected": "A"})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default="data/processed/rjokes", help="folder with {train,dev,test}.jsonl")
    ap.add_argument("--out", required=True, help="folder for dpo_{train,dev,test}.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        src = pathlib.Path(args.splits) / f"{split}.jsonl"
        rows = [json.loads(line) for line in open(src, encoding="utf-8")]
        records = build(rows, args.seed)
        with open(out_dir / f"dpo_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        print(f"{src}: {len(rows)} jokes -> {len(records)} records ({len(records) // 2} pairs)")


if __name__ == "__main__":
    main()

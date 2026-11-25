"""Step 1, option A: LLM-written premises for the joke pool (docs/plans/sft-dpo-ogrpo.md).

Usage (cluster, repo root):
    python -m explorer.gen_premises --out data/premises/premises-v1.jsonl --n 1150

Asks the teacher for short everyday premises spread over joke_prompts.AREAS, drops near-duplicates (case,
punctuation and a leading article ignored) and premises the live safety rules reject, and keeps the first --n.
Each premise gets a stable id (a hash of its normalized text) and a split: "eval" for about 1 in 5 (id-based, so
adding premises never moves an old one), "train" for the rest. Eval premises are never trained on and need no
candidates, only a prompt (step 5). 1 in 5 of --n 1150 gives ~230 eval premises, enough for a +-0.06 CI on a
paired win rate (sd ~0.45 per premise), and ~920 train premises (HumorGen used 1,200).
Written by our model, not taken from Reddit or transcripts, so the file can be committed after a hand check.
"""
import argparse
import hashlib
import math
import re

from explorer.joke_prompts import AREAS, parse_lines, premise_messages
from explorer.pool_io import write_jsonl

EVAL_EVERY = 5  # about 1 premise in 5 is held out for step 5


def normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    t = re.sub(r"^(the|a|an) ", "", " ".join(t.split()))
    return t


def premise_id(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:10]


def split_of(pid: str) -> str:
    return "eval" if int(pid, 16) % EVAL_EVERY == 0 else "train"


def collect(replies: list[tuple[str, str]], n: int, is_safe=lambda t: True) -> list[dict]:
    """(area, reply) pairs -> up to n premise rows, first occurrence wins, round-robin over areas so no area
    crowds out the others when n is reached."""
    by_area: dict[str, list[str]] = {}
    for area, reply in replies:
        by_area.setdefault(area, []).extend(parse_lines(reply))
    seen, rows = set(), []
    queues = {a: list(v) for a, v in by_area.items()}
    while len(rows) < n and any(queues.values()):
        for area, q in queues.items():
            while q:
                text = q.pop(0)
                key = normalize(text)
                if key in seen or not (2 <= len(key.split()) <= 10) or not is_safe(text):
                    continue
                seen.add(key)
                pid = premise_id(text)
                rows.append({"id": pid, "premise": text, "area": area, "split": split_of(pid)})
                break
            if len(rows) >= n:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/premises/premises-v1.jsonl")
    ap.add_argument("--n", type=int, default=1150)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--per-call", type=int, default=25, help="premises asked for per call")
    ap.add_argument("--overshoot", type=float, default=2.0, help="ask for this many times --n (dups get dropped)")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from explorer.hf_generate import make_generator
    from explorer.safety_rules import is_safe

    calls_per_area = math.ceil(args.n * args.overshoot / args.per_call / len(AREAS))
    jobs = [a for a in AREAS for _ in range(calls_per_area)]
    generate = make_generator(args.model, dtype=args.dtype, batch_size=args.batch_size, temperature=1.0,
                              max_new_tokens=600, seed=args.seed)
    replies = generate([premise_messages(a, args.per_call) for a in jobs])
    rows = collect(list(zip(jobs, replies)), args.n, is_safe)
    write_jsonl(args.out, rows)
    n_eval = sum(r["split"] == "eval" for r in rows)
    print(f"{len(jobs)} calls -> {len(rows)} premises ({n_eval} eval) in {args.out}")
    if len(rows) < args.n:
        print(f"short of --n {args.n}: raise --overshoot")


if __name__ == "__main__":
    main()

"""Step 0 gate: how often does the A/B judge agree with rJokes upvotes on held-out pairs?

Usage (cluster, repo root):
    python -m explorer.eval_judge --out runs/judge_eval/base.json                     # untrained baseline
    python -m explorer.eval_judge --adapter $COURSE_DIR/runs/dpo_judge_v1/final \
        --out runs/judge_eval/adapter.json --baseline runs/judge_eval/base.json      # trained + paired gate

Scored per pair, not per record: build_judge_pairs.py writes every pair in both orders, so the two records
of a pair aren't independent. A pair scores 1 if the judge is right in both orders, 0.5 if right in one
(order-inconsistent), 0 if wrong in both (DPO_training.md "Scoring"). The 95% CI is over pairs.
With --baseline, also reports the paired difference (this judge minus the baseline, same pairs) and the gate:
pass when this judge's CI lies above 0.5 and the paired difference's CI lies above 0 (sft-dpo-ogrpo.md).
Pairs are keyed by a hash of their two texts, so the result file holds no joke text.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re

PAIR = re.compile(r"Which joke is funnier\?\nJoke A: (.*)\nJoke B: (.*)\nAnswer with A or B\.$", re.S)


def pair_key(prompt: str) -> str:
    """The same key for both orders of a pair: a hash of the two joke texts, sorted."""
    m = PAIR.match(prompt)
    if not m:
        raise ValueError(f"not a judge prompt: {prompt[:80]!r}")
    return hashlib.sha1(json.dumps(sorted([m.group(1), m.group(2)])).encode()).hexdigest()[:16]


def first_pairs(rows: list[dict], n_pairs: int) -> list[dict]:
    """All records (both orders) of the first n_pairs pairs in file order. Cutting records instead would
    leave pairs with one order only, which can't show order inconsistency."""
    keep: set[str] = set()
    for r in rows:
        if len(keep) == n_pairs:
            break
        keep.add(pair_key(r["prompt"]))
    return [r for r in rows if pair_key(r["prompt"]) in keep]


def pair_scores(rows: list[dict], p_a: list[float]) -> dict[str, float]:
    """pair key -> mean correctness over the pair's records (1, 0.5 or 0 when both orders are present).
    A NaN score (a prompt too long for the GPU) counts as wrong."""
    hits: dict[str, list[int]] = {}
    for r, p in zip(rows, p_a):
        right = not math.isnan(p) and ("A" if p > 0.5 else "B") == r["chosen"]
        hits.setdefault(pair_key(r["prompt"]), []).append(int(right))
    return {k: sum(v) / len(v) for k, v in hits.items()}


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, m - z * sd / math.sqrt(n), m + z * sd / math.sqrt(n)


def summarize(rows: list[dict], p_a: list[float]) -> dict:
    scores = pair_scores(rows, p_a)
    acc, lo, hi = mean_ci(list(scores.values()))
    answered = [p for p in p_a if not math.isnan(p)]
    counts: dict[str, int] = {}
    for r in rows:
        counts[pair_key(r["prompt"])] = counts.get(pair_key(r["prompt"]), 0) + 1
    complete = [k for k, c in counts.items() if c == 2]
    return {"n_records": len(rows), "n_pairs": len(scores), "n_pairs_both_orders": len(complete),
            "accuracy": acc, "ci95": [lo, hi],
            # an always-"A" judge is right in exactly one order of every pair: consistent share 0, accuracy 0.5
            "consistent_share": sum(scores[k] != 0.5 for k in complete) / max(len(complete), 1),
            "share_answered_A": sum(p > 0.5 for p in answered) / max(len(answered), 1),
            "n_nan": len(p_a) - len(answered), "pair_scores": scores}


def paired_diff(this: dict[str, float], base: dict[str, float]) -> dict:
    """This judge minus the baseline, on the pairs both scored."""
    common = sorted(set(this) & set(base))
    d, lo, hi = mean_ci([this[k] - base[k] for k in common])
    return {"n_pairs": len(common), "diff": d, "ci95": [lo, hi]}


def gate(result: dict) -> bool:
    """Step 0 passes when the judge's CI is above chance and its paired gain over the baseline is above 0."""
    return result["ci95"][0] > 0.5 and result["vs_baseline"]["ci95"][0] > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="leave out for the untrained baseline")
    ap.add_argument("--data", default=f"/scratch/{os.environ.get('USER', 'me')}/dpo/dpo_test.jsonl")
    ap.add_argument("--n_pairs", type=int, default=-1, help="first n pairs only, both orders (-1 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None, help="result JSON of the untrained judge, for the paired gate")
    ap.add_argument("--dtype", default="auto", help="auto | fp16 | bf16 | fp32 (fp16 on a V100)")
    ap.add_argument("--fp16", action="store_true", help="old spelling of --dtype fp16")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from explorer.judge_model import load, make_scorer

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.n_pairs > 0:
        rows = first_pairs(rows, args.n_pairs)
    tok, model = load(args.base, args.adapter, "fp16" if args.fp16 else args.dtype)
    p_a = make_scorer(tok, model, args.batch_size)([r["prompt"] for r in rows])

    result = {"base": args.base, "adapter": args.adapter, "data": args.data, **summarize(rows, p_a)}
    if args.baseline:
        result["vs_baseline"] = paired_diff(result["pair_scores"], json.load(open(args.baseline))["pair_scores"])
        result["gate_pass"] = gate(result)
    print(json.dumps({k: v for k, v in result.items() if k != "pair_scores"}, indent=2))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**result, "p_a": p_a}, open(out, "w"))


if __name__ == "__main__":
    main()

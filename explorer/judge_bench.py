"""Measure a running judge server from inside the cluster: does it score like Exp 17, and how long does a bit take?

Usage (cluster; explorer/judge_bench.sbatch starts the server first):
    python -m explorer.judge_bench --url http://localhost:8765 --out runs/judge_bench.json

A bit needs 6 prompts (3 threads, every pair in both orders). Timed for two sizes: single jokes (as trained) and
5-joke threads (each side 5 test jokes joined), 100 requests each. Correctness: the first 500 test pairs through
the server, compared pair by pair with Exp 17's run (runs/judge_eval/adapter.json). Writes numbers only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

from explorer import eval_judge as ej

TEMPLATE = "Which joke is funnier?\nJoke A: {a}\nJoke B: {b}\nAnswer with A or B."


def post(url: str, prompts: list[str]) -> tuple[list[float], float, float]:
    body = json.dumps({"prompts": prompts}).encode()
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(urllib.request.Request(f"{url}/score", body, {"Content-Type": "application/json"}), timeout=60))
    return res["p_a"], res["seconds"], time.perf_counter() - t


def pct(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"p50": statistics.median(xs), "p95": xs[int(0.95 * (len(xs) - 1))], "max": xs[-1], "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--data", default="/scratch/tingyu.z/dpo/dpo_test.jsonl")
    ap.add_argument("--reference", default="runs/judge_eval/adapter.json")
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--out", default="runs/judge_bench.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    sub = ej.first_pairs(rows, args.n_pairs)
    p_a = []
    for i in range(0, len(sub), 32):
        p_a += post(args.url, [r["prompt"] for r in sub[i:i + 32]])[0]
    p_a = [float("nan") if p is None else p for p in p_a]
    mine = ej.summarize(sub, p_a)
    ref = json.load(open(args.reference))["pair_scores"]
    keys = list(mine["pair_scores"])
    same = sum(mine["pair_scores"][k] == ref[k] for k in keys) / len(keys)
    racc = sum(ref[k] for k in keys) / len(keys)

    jokes = [ej.PAIR.match(r["prompt"]).group(1) for r in rows[::2][:600]]
    def bit(size: int, i: int) -> list[str]:
        t = [" ".join(jokes[(i * 15 + k * 5 + j) % len(jokes)] for j in range(size)) for k in range(3)]
        return [TEMPLATE.format(a=t[a], b=t[b]) for a in range(3) for b in range(3) if a != b]
    timing = {}
    for name, size in (("single_jokes", 1), ("five_joke_threads", 5)):
        server, total = [], []
        for i in range(100):
            _, s, w = post(args.url, bit(size, i))
            server.append(s)
            total.append(w)
        timing[name] = {"server_s": pct(server), "request_s": pct(total)}
    out = {"n_pairs": mine["n_pairs"], "accuracy": mine["accuracy"], "reference_accuracy_same_pairs": racc,
           "pair_scores_identical_share": same, "consistent_share": mine["consistent_share"], "n_nan": mine["n_nan"],
           "bit_of_6_prompts": timing}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

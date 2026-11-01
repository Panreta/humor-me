"""Is the labeled sample big enough? Per-mode 95% CI of the mean upvote percentile.

Usage (laptop or cluster, no GPU):
    python explorer/mode_precision.py [data/processed/rjokes/modes.jsonl] [--max-halfwidth 0.02]

Rule fixed before labeling (explore-next #1): if any mode's 95% CI half-width exceeds
--max-halfwidth, extend the sample by 30k (same seed, larger --n) and check again. Precision of
a mean depends on jokes per mode, not on the share of rJokes labeled.
"""
from __future__ import annotations  # cluster venv is Python 3.9: keeps `X | None` hints unevaluated

import argparse
import collections
import json
import math
import pathlib
import sys

Z95 = 1.96


def per_mode(records: list[dict]) -> dict[str, dict]:
    by = collections.defaultdict(list)
    for r in records:
        if "mode" in r:  # parse failures carry "raw" instead
            by[r["mode"]].append(r["pct"])
    out = {}
    for mode, xs in sorted(by.items()):
        n, mean = len(xs), sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else float("inf")
        out[mode] = {"n": n, "mean_pct": mean, "halfwidth": Z95 * sd / math.sqrt(n)}
    return out


def run(path: str = "data/processed/rjokes/modes.jsonl", max_halfwidth: float = 0.02,
        first_n: int | None = None) -> dict:
    """first_n: the sample only grows by appending, so its prefix is the earlier, smaller run."""
    recs = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()][:first_n]
    stats = per_mode(recs)
    return {"labeled": len(recs), "failed": sum("raw" in r for r in recs), "modes": stats,
            "wide": [m for m, s in stats.items() if m != "none" and s["halfwidth"] > max_halfwidth]}


def table(res: dict) -> str:
    out = [f"{res['labeled']} labeled, {res['failed']} parse failures ({res['failed'] / max(1, res['labeled']):.1%})\n",
           "| Mode | Jokes | Mean pct | 95% CI ± |\n|---|---|---|---|"]
    for m, s in sorted(res["modes"].items(), key=lambda kv: -kv[1]["mean_pct"]):
        out.append(f"| {m} | {s['n']} | {s['mean_pct']:.3f} | {s['halfwidth']:.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/processed/rjokes/modes.jsonl")
    ap.add_argument("--max-halfwidth", type=float, default=0.02)
    a = ap.parse_args()
    res = run(a.path, a.max_halfwidth)
    print(table(res))
    if res["wide"]:
        print(f"\nToo wide (> ±{a.max_halfwidth}): {', '.join(res['wide'])} → extend: "
              f"rerun with --n {res['labeled'] + 30000}")
        sys.exit(1)
    print(f"\nAll modes within ±{a.max_halfwidth}: sample is big enough.")


if __name__ == "__main__":
    main()

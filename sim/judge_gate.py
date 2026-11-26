"""Exp 17: does the DPO-trained rJokes judge pick the more-upvoted joke, above chance and above the untrained model?

Usage (laptop, after copying the cluster's eval results back):
    python -m sim.judge_gate [--dir data/results/exp17_judge_eval] [--record]

Reads the two result files `explorer/eval_judge.py` wrote on Explorer (`base.json`: untrained Qwen2.5-7B-Instruct;
`adapter.json`: the same model with the `dpo_judge_v1` LoRA adapter), each holding a score per pair (1 right in both
orders, 0.5 right in one, 0 wrong in both) keyed by a hash of the two texts, so no joke text. Recomputes every number
from those per-pair scores rather than trusting the summaries, and checks the two agree. `--record` writes
`reports/metrics/judge_gate.json`: numbers only, no pair keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

from sim import cache

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "exp17_judge_eval"
NAME = "judge_gate"
N_TEST_PAIRS = 5318  # rJokes test split, every pair in both orders (explorer/build_judge_pairs.py)


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI, the same formula as explorer/eval_judge.py."""
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, m - z * sd / math.sqrt(n), m + z * sd / math.sqrt(n)


def outcomes(scores: dict[str, float]) -> dict:
    """How the judge did per pair: right in both orders, in one only (it flipped with the order), in neither."""
    both = sum(s == 1.0 for s in scores.values())
    one = sum(s == 0.5 for s in scores.values())
    neither = sum(s == 0.0 for s in scores.values())
    acc, lo, hi = mean_ci(list(scores.values()))
    return {"n_pairs": len(scores), "right_both": both, "right_one": one, "right_neither": neither,
            "accuracy": acc, "ci95": [lo, hi], "consistent_share": (both + neither) / len(scores),
            # When it gives the same answer in both orders, how often is that answer the more-upvoted joke?
            "accuracy_when_consistent": both / max(both + neither, 1)}


def gate(trained: dict, diff: dict) -> bool:
    """The rule fixed before the run (D-116): the trained judge's CI lies above 0.5 and its paired gain over the
    untrained model's CI lies above 0. Same rule as explorer/eval_judge.py gate()."""
    return trained["ci95"][0] > 0.5 and diff["ci95"][0] > 0


def digest(directory: pathlib.Path) -> str:
    h = hashlib.sha256()
    for name in ("base.json", "adapter.json"):
        h.update((directory / name).read_bytes())
    return h.hexdigest()[:12]


def run(directory: str | pathlib.Path = RESULTS, key: str = "") -> dict:
    d = pathlib.Path(directory)
    raw = {arm: json.loads((d / f"{f}.json").read_text(encoding="utf-8")) for arm, f in (("untrained", "base"), ("trained", "adapter"))}
    arms = {arm: outcomes(r["pair_scores"]) for arm, r in raw.items()}
    for arm, r in raw.items():
        arms[arm].update(share_answered_A=r["share_answered_A"], n_unscored=r["n_nan"], adapter=r["adapter"])
    common = sorted(set(raw["trained"]["pair_scores"]) & set(raw["untrained"]["pair_scores"]))
    deltas = [raw["trained"]["pair_scores"][k] - raw["untrained"]["pair_scores"][k] for k in common]
    m, lo, hi = mean_ci(deltas)
    diff = {"n_pairs": len(common), "mean": m, "ci95": [lo, hi],
            "trained_better": sum(x > 0 for x in deltas), "same": sum(x == 0 for x in deltas), "untrained_better": sum(x < 0 for x in deltas)}
    # The eval script's own summary must match the numbers recomputed here from its per-pair scores.
    t, b = raw["trained"], raw["untrained"]
    close = lambda x, y: abs(x - y) < 1e-9
    agrees = {
        "accuracy": close(t["accuracy"], arms["trained"]["accuracy"]) and close(b["accuracy"], arms["untrained"]["accuracy"]),
        "ci95": all(close(x, y) for x, y in zip(t["ci95"] + b["ci95"], arms["trained"]["ci95"] + arms["untrained"]["ci95"])),
        "consistent_share": close(t["consistent_share"], arms["trained"]["consistent_share"]) and close(b["consistent_share"], arms["untrained"]["consistent_share"]),
        "paired_diff": close(t["vs_baseline"]["diff"], m) and all(close(x, y) for x, y in zip(t["vs_baseline"]["ci95"], [lo, hi])),
        "gate_pass": t["gate_pass"] == gate(arms["trained"], diff),
        # Both orders of every pair are present, so the mean pair score equals the per-record accuracy.
        "both_orders_everywhere": t["n_pairs_both_orders"] == t["n_pairs"] == b["n_pairs_both_orders"] == b["n_pairs"],
    }
    return {"exp": 17, "data": "rJokes test pairs, both orders", "n_test_pairs": N_TEST_PAIRS, "base": t["base"],
            "arms": arms, "paired_diff": diff, "gate_pass": gate(arms["trained"], diff), "eval_script_agrees": agrees,
            "bar": "trained CI lower bound > 0.5 and paired-difference CI lower bound > 0 (D-116)"}


def record(directory: str | pathlib.Path = RESULTS) -> dict:
    d = pathlib.Path(directory)
    return cache.run_or_load(NAME, lambda key: run(d, key), rerun=True, key=digest(d))


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def table(res: dict) -> str:
    u, t, d = res["arms"]["untrained"], res["arms"]["trained"], res["paired_diff"]
    pct = lambda a, k: f"{a[k] / a['n_pairs']:.0%} ({a[k]:,})"
    rows = [("Accuracy, per pair (95% CI)", *(f"{a['accuracy']:.3f} ({a['ci95'][0]:.3f} to {a['ci95'][1]:.3f})" for a in (u, t))),
            ("Right in both orders", pct(u, "right_both"), pct(t, "right_both")),
            ("Right in one order only (flipped)", pct(u, "right_one"), pct(t, "right_one")),
            ("Wrong in both orders", pct(u, "right_neither"), pct(t, "right_neither")),
            ("Same answer both orders", f"{u['consistent_share']:.1%}", f"{t['consistent_share']:.1%}"),
            ("Right when consistent", f"{u['accuracy_when_consistent']:.1%}", f"{t['accuracy_when_consistent']:.1%}"),
            ("Share answered \"A\" (position bias)", f"{u['share_answered_A']:.1%}", f"{t['share_answered_A']:.1%}")]
    lines = ["| | Untrained Qwen2.5-7B | DPO judge (`dpo_judge_v1`) |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines.append(f"\nPaired gain, trained minus untrained, same {d['n_pairs']:,} pairs: **{d['mean']:+.3f}** "
                 f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}); trained better on {d['trained_better']:,} pairs, "
                 f"same on {d['same']:,}, worse on {d['untrained_better']:,}. Gate: **{'pass' if res['gate_pass'] else 'fail'}**.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(RESULTS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 17 evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.dir) if a.record else run(a.dir)
    print(table(res))
    print("eval script agrees:", res["eval_script_agrees"])
    print(f"recorded to reports/metrics/{NAME}.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()

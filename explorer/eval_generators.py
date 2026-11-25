"""Step 5: compare base, SFT, DPO and O-GRPO on held-out premises (docs/plans/sft-dpo-ogrpo.md).

Usage (cluster, repo root):
    # 1. each model writes one line per eval prompt (runs/sets/eval_prompts.jsonl, never trained on)
    python -m explorer.eval_generators generate --models base sft=$R/sft_v1/final dpo=$R/dpo_generator_v1/final \
        ogrpo=$R/ogrpo_v1/final --out runs/eval/lines.jsonl
    # 2. a judge that did NOT build the training sets compares every pair of models on every premise, both orders
    python -m explorer.eval_generators judge --lines runs/eval/lines.jsonl --out runs/eval/report.json

`generate`: a model is `name=adapter_dir`, or `base` for the plain base model. Greedy decoding, so a rerun gives
the same lines. Resumable per model.
`judge`: default judge Qwen2.5-32B-Instruct, prompted with the premise (judge.judges.LogitJudge.WITH_PREMISE),
scored from its A/B logits. For each premise and model pair, both orders are asked; the pair scores 1 for the
model that wins both, 0.5 each on a split. Reports
- paired win rate with a 95% CI over premises for each comparison in --compare (default: sft vs base, dpo vs
  sft, ogrpo vs sft, dpo vs ogrpo). A model is better only when its CI lies above 0.5 (CLAUDE.md rule 4);
- a Bradley-Terry leaderboard over models with bootstrap CIs (premises resampled);
- each model's share of lines the live safety rules reject.
If the judge is also the teacher (the 32B wrote the candidates), it may favour its own style: read the
comparisons between trained models, which share that bias, rather than any model vs the teacher.
"""
from __future__ import annotations  # cluster venv was Python 3.9: keeps `X | None` hints unevaluated

import argparse
import itertools
import json
import math
import pathlib
import random

from explorer.pool_io import append_jsonl, read_jsonl

DEFAULT_COMPARE = ("sft:base", "dpo:sft", "ogrpo:sft", "dpo:ogrpo")


def parse_models(specs: list[str]) -> dict[str, str | None]:
    out = {}
    for s in specs:
        name, _, adapter = s.partition("=")
        out[name] = adapter or None
    return out


def clean(reply: str) -> str:
    """The first non-empty line of a reply, quotes stripped."""
    for line in reply.strip().splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""


def by_premise(lines: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in lines:
        out.setdefault(r["premise_id"], {})[r["model"]] = r
    return out


def jobs(lines: list[dict]) -> list[tuple[str, str, str]]:
    """(premise_id, first model, second model) for every ordered model pair on every premise that has both."""
    out = []
    for pid, models in by_premise(lines).items():
        for a, b in itertools.permutations(sorted(models), 2):
            out.append((pid, a, b))
    return out


def pair_points(results: dict[tuple[str, str, str], float]) -> dict[tuple[str, str, str], float]:
    """(premise, a, b) with a < b -> points for a: 1 if a wins in both orders, 0.5 on a split, 0 if b wins both."""
    pts = {}
    for (pid, first, second), p in results.items():
        a, b = sorted((first, second))
        win_a = (p > 0.5) == (first == a)
        pts.setdefault((pid, a, b), []).append(1.0 if win_a else 0.0)
    return {k: sum(v) / len(v) for k, v in pts.items() if len(v) == 2}


def win_rate(points: dict[tuple[str, str, str], float], x: str, y: str, z: float = 1.96) -> dict:
    """x's paired win rate against y over premises, with a 95% CI."""
    s = [v if a == x else 1 - v for (pid, a, b), v in points.items() if {a, b} == {x, y}]
    if not s:
        return {"n": 0}
    m = sum(s) / len(s)
    sd = math.sqrt(sum((v - m) ** 2 for v in s) / (len(s) - 1)) if len(s) > 1 else 0.0
    half = z * sd / math.sqrt(len(s))
    verdict = "better" if m - half > 0.5 else "worse" if m + half < 0.5 else "inconclusive"
    return {"n": len(s), "win_rate": m, "ci95": [m - half, m + half], "verdict": verdict}


def leaderboard(points, models: list[str], boot: int = 1000, seed: int = 0) -> dict:
    """Bradley-Terry strength per model from the per-premise points, with bootstrap CIs over premises."""
    from judge.bt import fit_bt

    idx = {m: i for i, m in enumerate(models)}
    premises = sorted({pid for pid, _, _ in points})

    def fit(sample):
        games = []
        for pid in sample:
            for (p, a, b), v in points.items():
                if p == pid:
                    games += [(idx[a], idx[b])] * round(2 * v) + [(idx[b], idx[a])] * round(2 * (1 - v))
        return fit_bt(len(models), games)[0]

    theta = fit(premises)
    rng = random.Random(seed)
    draws = [fit([rng.choice(premises) for _ in premises]) for _ in range(boot)]
    out = {}
    for m, i in idx.items():
        vals = sorted(d[i] for d in draws)
        out[m] = {"theta": float(theta[i]), "ci95": [float(vals[int(0.025 * boot)]), float(vals[int(0.975 * boot) - 1])]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["theta"]))


def cmd_generate(args) -> None:
    from explorer.hf_generate import make_generator

    prompts = read_jsonl(args.prompts)
    done = {(r["model"], r["premise_id"]) for r in read_jsonl(args.out)}
    for name, adapter in parse_models(args.models).items():
        todo = [p for p in prompts if (name, p["premise_id"]) not in done]
        if not todo:
            continue
        gen = make_generator(args.base, adapter, args.dtype, args.batch_size, temperature=0, max_new_tokens=80)
        replies = gen([p["prompt"] for p in todo])
        append_jsonl(args.out, [{"model": name, "adapter": adapter, "premise_id": p["premise_id"],
                                 "premise": p["premise"], "mode": p["mode"], "text": clean(r)}
                                for p, r in zip(todo, replies)])
        print(f"{name}: {len(todo)} lines", flush=True)
        del gen


def cmd_judge(args) -> None:
    from explorer.judge_model import load, make_scorer
    from explorer.safety_rules import is_safe
    from judge.judges import LogitJudge

    lines = read_jsonl(args.lines)
    table = by_premise(lines)
    work = jobs(lines)
    tok, model = load(args.judge, None, args.dtype)
    judge = LogitJudge(make_scorer(tok, model, args.batch_size), name=f"logit:{args.judge}",
                       template=LogitJudge.WITH_PREMISE)
    prompts = [judge.message(table[pid][a]["premise"], table[pid][a]["text"], table[pid][b]["text"])
               for pid, a, b in work]
    results = dict(zip(work, judge.score(prompts)))
    points = pair_points(results)
    models = sorted({r["model"] for r in lines})
    report = {
        "judge": judge.name, "premises": len(table), "calls": len(work),
        "first_shown_win_share": sum(p > 0.5 for p in results.values()) / max(len(results), 1),
        "compare": {c: win_rate(points, *c.split(":")) for c in args.compare},
        "leaderboard": leaderboard(points, models, args.boot),
        "unsafe_share": {m: sum(not is_safe(r["text"]) for r in lines if r["model"] == m) /
                         max(sum(r["model"] == m for r in lines), 1) for m in models},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--models", nargs="+", required=True, help="base and/or name=adapter_dir")
    g.add_argument("--prompts", default="runs/sets/eval_prompts.jsonl")
    g.add_argument("--out", default="runs/eval/lines.jsonl")
    g.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    g.add_argument("--dtype", default="auto")
    g.add_argument("--batch-size", type=int, default=32)
    j = sub.add_parser("judge")
    j.add_argument("--lines", default="runs/eval/lines.jsonl")
    j.add_argument("--out", default="runs/eval/report.json")
    j.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
    j.add_argument("--compare", nargs="+", default=list(DEFAULT_COMPARE), help="x:y = x's win rate against y")
    j.add_argument("--boot", type=int, default=1000)
    j.add_argument("--dtype", default="auto")
    j.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    cmd_generate(args) if args.cmd == "generate" else cmd_judge(args)


if __name__ == "__main__":
    main()

"""Step 3: turn the ranked pool into the SFT, DPO and O-GRPO training sets (HumorGen 5.1, Appendix H).

Usage (laptop or cluster, repo root):
    python -m explorer.build_sets --ranked runs/pool/ranked.jsonl --out runs/sets

From each train premise's lines, sorted by the judge's Bradley-Terry theta:
- SFT: the top --sft-top lines (HumorGen: 10), each with its own mode in the prompt.
- DPO: with the mode in the prompt (default), one pair per mode: that mode's best line (chosen) vs its worst
  (rejected), when their theta differ. A pair shares one prompt, so both lines must be in the same mode; the
  global top-5 vs bottom-5 pairing would mix modes. With --no-mode the prompt has the premise only and pairs
  are HumorGen's: --dpo-pairs random top-5 vs bottom-5 matches.
- O-GRPO: every line, with advantage A = (theta - group mean) / (group sd + 1e-6) over the premise's lines and
  weight w = softmax(A / T) over the group (sums to 1). Stored as weight = w * G, so a plain mean over
  shuffled examples equals HumorGen's E_x[sum_i w_i * loss_i].
Premises whose lines all have the same theta (a judge with no preference, e.g. one that always answers "A")
carry no ranking and are dropped. About 5% of train premises (by id) are held out as dev, for early stopping.
Eval premises go to eval_prompts.jsonl only (step 5), one mode each, cycling over the six.
Writes {sft,dpo,ogrpo}_{train,dev}.jsonl, eval_prompts.jsonl and stats.json to --out.
"""
import argparse
import json
import math
import pathlib
import random
import statistics

from explorer.joke_prompts import MODES, PROMPT_ID, student_messages
from explorer.pool_io import read_jsonl, write_jsonl

DEV_EVERY = 20  # about 1 train premise in 20 is dev
EPS = 1e-6


def is_dev(premise_id: str) -> bool:
    return int(premise_id, 16) % DEV_EVERY == 1  # eval premises are the ids with remainder 0 mod 10


def answer(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def informative(row: dict) -> bool:
    thetas = [l["theta"] for l in row["lines"]]
    return max(thetas) - min(thetas) > 1e-9


def sft_rows(row: dict, top: int, use_mode: bool) -> list[dict]:
    return [{"prompt": student_messages(row["premise"], l["mode"] if use_mode else None), "completion": answer(l["text"]),
             "premise_id": row["premise_id"], "mode": l["mode"], "theta": l["theta"]}
            for l in row["lines"][:top]]


def dpo_rows(row: dict, use_mode: bool, n_pairs: int, rng: random.Random) -> list[dict]:
    lines = row["lines"]  # best first
    pairs = []
    if use_mode:
        for mode in MODES:
            ms = [l for l in lines if l["mode"] == mode]
            if len(ms) >= 2 and ms[0]["theta"] > ms[-1]["theta"]:
                pairs.append((mode, ms[0], ms[-1]))
    else:
        k = min(5, len(lines) // 2)
        top, bottom = lines[:k], lines[-k:]
        for _ in range(n_pairs):
            c, r = rng.choice(top), rng.choice(bottom)
            if c["theta"] > r["theta"]:
                pairs.append((None, c, r))
    return [{"prompt": student_messages(row["premise"], mode), "chosen": answer(c["text"]), "rejected": answer(r["text"]),
             "premise_id": row["premise_id"], "mode": mode, "theta_gap": c["theta"] - r["theta"]}
            for mode, c, r in pairs]


def advantages(thetas: list[float], temperature: float = 1.0) -> tuple[list[float], list[float]]:
    """(A, w): group z-scores and their softmax weights (HumorGen eq. 2-4)."""
    mu = statistics.fmean(thetas)
    sd = statistics.pstdev(thetas)
    a = [(t - mu) / (sd + EPS) for t in thetas]
    m = max(a)
    e = [math.exp((x - m) / temperature) for x in a]
    z = sum(e)
    return a, [x / z for x in e]


def ogrpo_rows(row: dict, use_mode: bool, temperature: float) -> list[dict]:
    lines = row["lines"]
    a, w = advantages([l["theta"] for l in lines], temperature)
    g = len(lines)
    return [{"prompt": student_messages(row["premise"], l["mode"] if use_mode else None), "completion": answer(l["text"]),
             "premise_id": row["premise_id"], "mode": l["mode"], "advantage": ai, "weight": wi * g}
            for l, ai, wi in zip(lines, a, w)]


def build(ranked: list[dict], *, sft_top: int = 10, use_mode: bool = True, dpo_pairs: int = 5,
          temperature: float = 1.0, seed: int = 0) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    sets = {f"{s}_{p}": [] for s in ("sft", "dpo", "ogrpo") for p in ("train", "dev")}
    sets["eval_prompts"] = []
    flat, modes = 0, list(MODES)
    for row in ranked:
        if row["split"] == "eval":
            mode = modes[int(row["premise_id"], 16) % len(modes)]
            sets["eval_prompts"].append({"premise_id": row["premise_id"], "premise": row["premise"],
                                         "mode": mode if use_mode else None,
                                         "prompt": student_messages(row["premise"], mode if use_mode else None)})
            continue
        if not informative(row):
            flat += 1
            continue
        part = "dev" if is_dev(row["premise_id"]) else "train"
        sets[f"sft_{part}"] += sft_rows(row, sft_top, use_mode)
        sets[f"dpo_{part}"] += dpo_rows(row, use_mode, dpo_pairs, rng)
        sets[f"ogrpo_{part}"] += ogrpo_rows(row, use_mode, temperature)
    for name in sets:
        if name.endswith("_train"):
            rng.shuffle(sets[name])

    train_rows = [r for r in ranked if r["split"] != "eval"]
    n_lines = sum(len(r["lines"]) for r in ranked)
    n_unsafe = sum(len(r["unsafe"]) for r in ranked)
    stats = {"prompt_id": PROMPT_ID, "mode_in_prompt": use_mode, "sft_top": sft_top, "temperature": temperature,
             "premises": len(ranked), "train_premises": len(train_rows), "eval_premises": len(sets["eval_prompts"]),
             "dropped_flat_premises": flat, "lines_ranked": n_lines, "lines_unsafe": n_unsafe,
             "unsafe_share": n_unsafe / max(n_lines + n_unsafe, 1),
             "judge_consistent_share_median": statistics.median([r["consistent_share"] for r in ranked]) if ranked else None,
             "judge_first_shown_win_share_mean": statistics.fmean([r["first_shown_win_share"] for r in ranked]) if ranked else None,
             "sft_mode_mix": {m: sum(r["mode"] == m for r in sets["sft_train"]) for m in MODES},
             "ogrpo_median_advantage": statistics.median([r["advantage"] for r in sets["ogrpo_train"]]) if sets["ogrpo_train"] else None,
             **{f"n_{k}": len(v) for k, v in sets.items()}}
    return sets, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ranked", default="runs/pool/ranked.jsonl")
    ap.add_argument("--out", default="runs/sets")
    ap.add_argument("--sft-top", type=int, default=10)
    ap.add_argument("--no-mode", action="store_true", help="premise-only prompts and HumorGen's global DPO pairs")
    ap.add_argument("--dpo-pairs", type=int, default=5, help="--no-mode only: pairs per premise")
    ap.add_argument("--temperature", type=float, default=1.0, help="O-GRPO softmax temperature T")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets, stats = build(read_jsonl(args.ranked), sft_top=args.sft_top, use_mode=not args.no_mode,
                        dpo_pairs=args.dpo_pairs, temperature=args.temperature, seed=args.seed)
    out = pathlib.Path(args.out)
    for name, rows in sets.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

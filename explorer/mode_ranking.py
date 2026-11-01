"""Do people rank the six modes the way HumorGen's LLM judge does? rJokes upvotes vs Table 4.

Usage (laptop, no GPU):
    python explorer/mode_ranking.py [--boots 1000]

Design and pass criteria were fixed before running (decisions.md 2026-11-01). Joke length predicts
upvotes and differs by mode, so every estimate compares jokes of similar length from the same year:
- WR: win rate vs the field, within year x length-bin cells (ties half). HumorGen's unit.
- REG: OLS pct ~ mode (sum-to-zero) + year + length bin. Effect in pct units vs the average mode.
Each with quartile and decile length bins: 4 variants. A claim is a finding only if all 4 agree.
"""
from __future__ import annotations  # cluster venv is Python 3.9: keeps `X | None` hints unevaluated

import argparse
import json
import pathlib

import numpy as np

# HumorGen Table 4: each persona's win rate in the LLM judge's pairwise comparisons.
JUDGE = {"neurotic": 0.634, "absurdist": 0.558, "cynic": 0.552, "observer": 0.491,
         "optimist": 0.411, "wordsmith": 0.349}
MODES = tuple(JUDGE)
VARIANTS = {"WR-q4": ("wr", 4), "WR-d10": ("wr", 10), "REG-q4": ("reg", 4), "REG-d10": ("reg", 10)}
MODES_PATH = "data/processed/rjokes/modes.jsonl"
JOKES_PATH = "data/processed/rjokes/train.jsonl"


def load(modes: str = MODES_PATH, jokes: str = JOKES_PATH) -> dict:
    lab = {}
    for line in pathlib.Path(modes).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("mode") in JUDGE:  # drops parse failures and "none" (no judge value)
            lab[r["id"]] = MODES.index(r["mode"])
    rows = []
    with open(jokes, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in lab and r["year"] is not None:  # ~0.3% of rJokes has no date: can't year-match
                rows.append((lab[r["id"]], r["pct"], r["year"], r["n_words"]))
    a = np.array(rows, dtype=float)
    return {"mode": a[:, 0].astype(int), "pct": a[:, 1], "year": a[:, 2].astype(int), "words": a[:, 3]}


def length_bins(words: np.ndarray, k: int) -> np.ndarray:
    cuts = np.quantile(words, np.arange(1, k) / k)
    return np.searchsorted(cuts, words, side="left")


def midranks(group: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1-based rank of each x within its group, ties sharing the average rank."""
    order = np.lexsort((x, group))
    g, v = group[order], x[order]
    pos = np.arange(len(x))
    new_group = np.r_[True, g[1:] != g[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, pos, 0))
    new_run = new_group | np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(new_run) - 1
    first = pos[new_run]
    last = np.r_[first[1:], len(x)] - 1
    mid = (first[run_id] + last[run_id]) / 2 - group_start + 1
    out = np.empty(len(x))
    out[order] = mid
    return out


def win_rates(mode: np.ndarray, cell: np.ndarray, pct: np.ndarray, n_modes: int = len(MODES)) -> np.ndarray:
    """Share of same-cell jokes from *other* modes that each mode's jokes beat (ties count half).
    Rank among everyone minus rank among your own mode = wins against the others."""
    sub = cell * n_modes + mode
    wins = midranks(cell, pct) - midranks(sub, pct)
    n_cell = np.bincount(cell)
    n_sub = np.bincount(sub, minlength=(cell.max() + 1) * n_modes)
    others = n_cell[np.arange(len(n_sub)) // n_modes] - n_sub
    pairs = np.bincount(np.arange(len(n_sub)) % n_modes, weights=n_sub * others, minlength=n_modes)
    return np.bincount(mode, weights=wins, minlength=n_modes) / pairs


def reg_effects(mode: np.ndarray, year: np.ndarray, lbin: np.ndarray, pct: np.ndarray,
                n_modes: int = len(MODES)) -> np.ndarray:
    """OLS mode effects vs the average mode (sum-to-zero), adjusted for year and length bin."""
    def dummies(v):
        levels = np.unique(v)[1:]
        return (v[:, None] == levels[None, :]).astype(float)
    m = np.zeros((len(mode), n_modes - 1))
    for j in range(n_modes - 1):
        m[:, j] = (mode == j).astype(float) - (mode == n_modes - 1)
    X = np.hstack([np.ones((len(mode), 1)), m, dummies(year), dummies(lbin)])
    beta = np.linalg.lstsq(X, pct, rcond=None)[0][1:n_modes]
    return np.r_[beta, -beta.sum()]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def estimate(kind: str, d: dict, lbin: np.ndarray, idx: np.ndarray) -> np.ndarray:
    mode, pct, year, lb = d["mode"][idx], d["pct"][idx], d["year"][idx], lbin[idx]
    if kind == "wr":
        _, cell = np.unique(year * 100 + lb, return_inverse=True)
        return win_rates(mode, cell.ravel(), pct)
    return reg_effects(mode, year, lb, pct)


def run(modes: str = MODES_PATH, jokes: str = JOKES_PATH, boots: int = 1000, seed: int = 0) -> dict:
    d = load(modes, jokes)
    n = len(d["pct"])
    judge = np.array([JUDGE[m] for m in MODES])
    rng = np.random.default_rng(seed)
    resamples = [rng.integers(0, n, n) for _ in range(boots)]
    everyone = np.arange(n)

    raw = np.array([d["pct"][d["mode"] == j].mean() for j in range(len(MODES))])
    out = {"n": n, "modes": list(MODES), "judge": judge.tolist(),
           "median_words": [float(np.median(d["words"][d["mode"] == j])) for j in range(len(MODES))],
           "raw_mean_pct": raw.tolist(), "raw_rho": spearman(raw, judge), "variants": {}}
    for name, (kind, k) in VARIANTS.items():
        lbin = length_bins(d["words"], k)
        point = estimate(kind, d, lbin, everyone)
        bs = np.array([estimate(kind, d, lbin, idx) for idx in resamples])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rhos = np.array([spearman(b, judge) for b in bs])
        rho_lo, rho_hi = np.percentile(rhos, [2.5, 97.5])
        mid = 0.5 if kind == "wr" else 0.0
        side = ["above" if l > mid else "below" if h < mid else "undetermined" for l, h in zip(lo, hi)]
        judge_side = ["above" if j > 0.5 else "below" for j in judge]
        out["variants"][name] = {
            "estimate": point.tolist(), "ci": [lo.tolist(), hi.tolist()],
            "rho": spearman(point, judge), "rho_ci": [float(rho_lo), float(rho_hi)],
            "c1": "agree" if rho_lo > 0 else "disagree" if rho_hi < 0 else "inconclusive",
            "side": side, "judge_side": judge_side,
            "c2_matches": sum(s == j for s, j in zip(side, judge_side)),
            "c2_undetermined": side.count("undetermined"),
            "wordsmith_last_share": float(np.mean(bs.argmin(axis=1) == MODES.index("wordsmith"))),
        }
    v = out["variants"].values()
    out["findings"] = {
        "c1": next(iter(v))["c1"] if len({x["c1"] for x in v}) == 1 else "depends on the estimator",
        "c2_side_by_mode": {m: (sides.pop() if len(sides := {x["side"][j] for x in v}) == 1
                                else "depends on the estimator") for j, m in enumerate(MODES)},
        "c3_wordsmith_last": all(x["wordsmith_last_share"] >= 0.95 for x in v),
    }
    return out


def table(res: dict) -> str:
    modes, judge = res["modes"], res["judge"]
    head = "| Mode | Median words | Judge win rate | Raw mean pct | " + " | ".join(res["variants"]) + " |"
    out = [head, "|" + "---|" * (4 + len(res["variants"]))]
    for j, m in enumerate(modes):
        cells = []
        for name, v in res["variants"].items():
            fmt = "{:.3f}" if name.startswith("WR") else "{:+.3f}"
            cells.append(f"{fmt.format(v['estimate'][j])} [{fmt.format(v['ci'][0][j])}, {fmt.format(v['ci'][1][j])}]")
        out.append(f"| {m} | {res['median_words'][j]:.0f} | {judge[j]:.1%} | {res['raw_mean_pct'][j]:.3f} | "
                   + " | ".join(cells) + " |")
    out.append("\nWR = win rate vs same-year, similar-length jokes of other modes; REG = effect on pct vs the "
               "average mode. [95% bootstrap CI]\n")
    out.append("| Variant | ρ vs judge [95% CI] | C1 | C2 sides matching (undetermined) | C3 wordsmith last |")
    out.append("|---|---|---|---|---|")
    out.append(f"| raw means | {res['raw_rho']:+.2f} | (not a criterion) | | |")
    for name, v in res["variants"].items():
        out.append(f"| {name} | {v['rho']:+.2f} [{v['rho_ci'][0]:+.2f}, {v['rho_ci'][1]:+.2f}] | {v['c1']} | "
                   f"{v['c2_matches']}/6 ({v['c2_undetermined']}) | {v['wordsmith_last_share']:.0%} of resamples |")
    f = res["findings"]
    sides = ", ".join(f"{m} {s}" for m, s in f["c2_side_by_mode"].items())
    out.append(f"\n**Across all 4 variants:** C1 {f['c1']}; C3 wordsmith last: "
               f"{'holds' if f['c3_wordsmith_last'] else 'not shown'}; human side of the field: {sides}.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=1000)
    a = ap.parse_args()
    print(table(run(boots=a.boots)))


if __name__ == "__main__":
    main()

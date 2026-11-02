"""Two tastes: does the director choice from Exp 4 hold if the room's taste comes from people?

Usage:
    python -m sim.evaluate_tastes

Criteria were fixed before running (decisions.md 2026-11-01). Only `base` changes between tastes: venue, age
and dynamics stay as they are, so any difference comes from how strongly the room prefers some modes.
"""
import copy
import json
import pathlib

import numpy as np

from sim.env import AudienceEnv, load_audience
from sim.evaluate_policies import ALGOS, COLD, Oracle, WARM, career, ci
from sim.policies import Fixed, LinUCB, Random, prior_means

WIN_RATES = pathlib.Path(__file__).resolve().parent.parent / "reports" / "metrics" / "mode_ranking.json"
CHOSEN = "LinUCB | uniform"


def base_from_win_rates(win_rates: dict) -> dict:
    """Same linear map as the judge's `base` in audience.yaml."""
    return {m: round(0.3 + 2 * (w - 0.5), 2) for m, w in win_rates.items()}


def human_win_rates(path: pathlib.Path = WIN_RATES) -> dict:
    res = json.loads(path.read_text(encoding="utf-8"))["results"]
    return dict(zip(res["modes"], res["variants"]["WR-d10"]["estimate"]))


def makers(modes: tuple) -> dict:
    judge_env = AudienceEnv()
    m = {"oracle": lambda env, s: Oracle(env), "random": lambda env, s: Random(len(modes), seed=s)}
    for j, mode in enumerate(modes):
        m[f"fixed {mode}"] = (lambda k: lambda env, s: Fixed(k))(j)
    for name, cls in ALGOS.items():
        m[f"{name} | uniform"] = (lambda c: lambda env, s: c(prior_means(env, "uniform", seed=s), seed=s))(cls)
    m["LinUCB | judge prior"] = lambda env, s: LinUCB(prior_means(judge_env, "humorgen", seed=s), seed=s)
    return m


def apart(a: list, b: list) -> bool:
    """True if a's CI sits entirely above b's."""
    return a[0] - a[1] > b[0] + b[1]


def criteria(rows: dict) -> dict:
    chosen, oracle = rows[CHOSEN]["warm"], rows["oracle"]["warm"]
    fixed = {k: v["warm"] for k, v in rows.items() if k.startswith("fixed ")}
    best = max(fixed, key=lambda k: fixed[k][0])
    others = [f"{a} | uniform" for a in ALGOS if f"{a} | uniform" != CHOSEN]
    beaten_by = [o for o in others if apart(rows[o]["warm"], chosen)]
    return {"share_of_oracle": chosen[0] / oracle[0], "p1": chosen[0] / oracle[0] >= 0.95,
            "best_fixed": best, "p2": apart(chosen, fixed[best]),
            "beaten_by": beaten_by, "p3": not beaten_by}


def tastes() -> dict:
    judge = load_audience()
    human = copy.deepcopy(judge)
    human["base"] = base_from_win_rates(human_win_rates())
    return {"judge": judge, "human": human}


def run(shows: int = 300, seeds: int = 10) -> dict:
    audiences = tastes()
    judge = audiences["judge"]
    out = {"base": {t: a["base"] for t, a in audiences.items()}, "tastes": {}}
    for taste, audience in audiences.items():
        rows = {}
        for name, make in makers(tuple(judge["modes"])).items():
            runs = [career(make, shows, seed, audience=audience) for seed in range(seeds)]
            laughs, lost = np.stack([r[0] for r in runs]), np.stack([r[1] for r in runs])
            rows[name] = {"cold": ci(laughs[:, :COLD].mean(1)), "warm": ci(laughs[:, -WARM:].mean(1)),
                          "lost_total": ci(lost.sum(1))}
        out["tastes"][taste] = {"rows": rows, "criteria": criteria(rows)}
    return out


def table(res: dict) -> str:
    out = [f"| Policy | Judge taste, last {WARM} shows | Laughs lost | Human taste, last {WARM} shows | Laughs lost |",
           "|---|---|---|---|---|"]
    for name, j in res["tastes"]["judge"]["rows"].items():
        h = res["tastes"]["human"]["rows"][name]
        out.append(f"| {name} | {j['warm'][0]:.1f} ± {j['warm'][1]:.1f} | {j['lost_total'][0]:.0f} ± {j['lost_total'][1]:.0f} | "
                   f"{h['warm'][0]:.1f} ± {h['warm'][1]:.1f} | {h['lost_total'][0]:.0f} ± {h['lost_total'][1]:.0f} |")
    return "\n".join(out)


if __name__ == "__main__":
    res = run()
    print(table(res))
    print({t: v["criteria"] for t, v in res["tastes"].items()})

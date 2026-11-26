"""Exp 31(b) (D-115): how often is the guess of the next bit's mode, made after joke 3, the mode the director picks?

Usage (at most 4 workers; set OPENBLAS_NUM_THREADS=1):
    python -m sim.evaluate_predict

Deciding at joke 3 failed Exp 31(a) (sim/evaluate_early.py): being one update behind costs about 1.3 laughs per show.
This design decides as before, at the bit's end, and only guesses at joke 3 (`app.predict.predict_mode`) so the next
bit can be written while the last two jokes play. A hit means the prep is ready when the bit ends; a miss throws it
away. The reward cannot change (the guess works on a copy; tests/test_predict_next.py runs the show with and without
it and compares the log), so only the hit rate is measured. A hit is "guess equals the mode picked", including the
picks that exploring (eps = 0.2, uniform over 6 modes) makes at random, which no guess can foresee: the ceiling is
about 0.8 + 0.2/6 = 0.83.

Bar, fixed before running: in every room variant at the 5-show horizon the 95% CI of the hit rate has its lower end at
least 0.5 (below that, most gaps stay and wasted calls dominate). Two guesses are compared, `learn` (the copy is first
taught the partial reward) and `now` (the model as it is). The one with the higher hit rate in "judge, gaps x1" at 5
shows is the one shipped and the one the bar is applied to; both are reported everywhere. Same ten rooms as Exp 31(a):
`base` gaps (x0.5, x1, x2; judge and human taste), `noise` (x0.5, x2) and `fatigue` (0, 0.3). The 30-show horizon is
reported and does not decide. Estimated seconds saved per show = hit rate x prep seconds x 7 gaps (an estimate: the
10 s prep is from show 0e102d77d795, not measured here).
Writes reports/metrics/early_guess.json (cached by params). Importable: run(), table(), criteria(), plot().
"""
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.director import EPSILON  # noqa: E402
from app.predict import predict_mode  # noqa: E402
from sim.cache import run_or_load  # noqa: E402
from sim.env import AudienceEnv  # noqa: E402
from sim.evaluate_early import variants  # noqa: E402
from sim.evaluate_epsilon import EXPLORE_SEED  # noqa: E402
from sim.evaluate_policies import CONTEXTS, ci  # noqa: E402
from sim.policies import EpsilonSoft, LinUCB, prior_means  # noqa: E402

GUESS_AFTER = 3
BAR = 0.5
HORIZONS = ((5, 1000), (30, 100))  # (shows, seeds)
GUESSES = ("learn", "now")
GATE_ROOM = "judge, gaps x1"
PREP_S, GAPS = 10.0, 7  # the estimate's inputs: prep seconds (show 0e102d77d795) and bit boundaries in a show of 8
WORKERS = 4  # a 16-process run exhausted memory


def career(audience: dict, shows: int, seed: int) -> dict:
    """Hits per guess rule over one career, and how often the guess equals the greedy mode (what a guess can foresee)."""
    env = AudienceEnv(audience)
    policy = EpsilonSoft(LinUCB(prior_means(env, "uniform", seed=seed), seed=seed), EPSILON, env.n_modes,
                         seed=seed + EXPLORE_SEED)
    ctx_rng = np.random.default_rng(seed)
    hits = {g: 0 for g in GUESSES}
    greedy_hits = {g: 0 for g in GUESSES}
    n = laughs = 0
    for s in range(shows):
        obs = env.reset(CONTEXTS[ctx_rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
        action = policy.decide(obs)[0]
        done = False
        while not done:
            nxt, reward, done, info = env.step(action)
            laughs += sum(info["levels"])
            guess = None
            if not done:
                guess = {g: predict_mode(policy.inner, obs, action, info["levels"][:GUESS_AFTER], n_bits=env.n_bits,
                                         learn=g == "learn") for g in GUESSES}
            policy.update(obs, action, reward)
            if not done:
                chosen, greedy, _ = policy.decide(nxt)
                n += 1
                for g in GUESSES:
                    hits[g] += guess[g] == chosen
                    greedy_hits[g] += guess[g] == greedy
                obs, action = nxt, chosen
    return {"n": n, "laughs": laughs / shows, **{f"hit_{g}": hits[g] / n for g in GUESSES},
            **{f"greedy_{g}": greedy_hits[g] / n for g in GUESSES}}


def _job(args):
    return career(*args)


def paired(audience: dict, shows: int, seeds: int, pool) -> dict:
    out = list(pool.map(_job, [(audience, shows, seed) for seed in range(seeds)], chunksize=20))
    res = {}
    for g in GUESSES:
        hit = np.array([o[f"hit_{g}"] for o in out])
        res[g] = {"hit": ci(hit), "greedy_match": ci(np.array([o[f"greedy_{g}"] for o in out])),
                  "seconds_saved_per_show": float(hit.mean() * PREP_S * GAPS)}
    res["laughs"] = ci(np.array([o["laughs"] for o in out]))
    return res


def run(horizons: list = [list(h) for h in HORIZONS]) -> dict:
    out = {"variants": {}}
    with ProcessPoolExecutor(WORKERS) as pool:
        for name, audience in variants().items():
            out["variants"][name] = {str(shows): paired(audience, shows, seeds, pool) for shows, seeds in horizons}
            print(name, flush=True)
    out["criteria"] = criteria(out)
    return out


def criteria(res: dict) -> dict:
    """The shipped guess is the better one in the gate room at 5 shows; it passes a room if its CI's lower end is at least BAR."""
    five = res["variants"][GATE_ROOM]["5"]
    shipped = max(GUESSES, key=lambda g: five[g]["hit"][0])
    rows = {}
    for name, by_horizon in res["variants"].items():
        mean, half = by_horizon["5"][shipped]["hit"]
        rows[name] = {"low": mean - half, "pass": bool(mean - half >= BAR)}
    return {"shipped": shipped, "variants": rows, "all_pass": all(r["pass"] for r in rows.values())}


def table(res: dict) -> str:
    out = ["| Room | Shows | Guess | Hit rate (95% CI) | Guess = greedy mode | Seconds saved per show (estimate) | Passes the bar |",
           "|---|---|---|---|---|---|---|"]
    for name, by_horizon in res["variants"].items():
        for shows, r in by_horizon.items():
            for g in GUESSES:
                verdict = ""
                if shows == "5" and g == res["criteria"]["shipped"]:
                    c = res["criteria"]["variants"][name]
                    verdict = f"{'yes' if c['pass'] else 'NO'} (low end {c['low']:.2f})"
                out.append(f"| {name} | {shows} | {g} | {r[g]['hit'][0]:.2f} ± {r[g]['hit'][1]:.2f} | "
                           f"{r[g]['greedy_match'][0]:.2f} | {r[g]['seconds_saved_per_show']:.0f} | {verdict} |")
    return "\n".join(out)


def plot(res: dict):
    """Hit rate at 5 shows per room for the shipped guess, with the bar."""
    import matplotlib.pyplot as plt

    g = res["criteria"]["shipped"]
    names = list(res["variants"])
    d = [res["variants"][n]["5"][g]["hit"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(names))
    ax.errorbar([m for m, _ in d], y, xerr=[h for _, h in d], fmt="o", color="#2a78d6", capsize=3)
    ax.axvline(BAR, color="#eb6834", ls="--", label=f"bar: lower end at least {BAR:g}")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"hit rate of the '{g}' guess at 5 shows (95% CI over seeds)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run_or_load("early_guess", run, horizons=[list(h) for h in HORIZONS])
    print(table(res))
    print(res["criteria"])

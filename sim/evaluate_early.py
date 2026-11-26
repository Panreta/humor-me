"""Exp 31 (D-115): does deciding the next bit's mode after joke 3 of the current bit cost reward?

Usage:
    python -m sim.evaluate_early

The live show can hide most of a bit's prep wait only by deciding the next mode before the current bit ends. That
decision sees a partial reward (the first `after` jokes, `reward_v1` on them) and is made one director update behind:
bit N+1 is decided before bit N's update, which still happens at bit N's end with the full reward. This sim plays
both orders on the same careers (same contexts and environment seeds, same exploration seeds) so the cost is a
paired difference per seed. Director: LinUCB under the live eps = 0.2 wrapper, uniform prior, as `app.director`.

The bar is fixed before running (CLAUDE.md, measure before cut): non-inferiority at the 5-show horizon the demo has.
For every room variant, the lower end of the 95% CI of (early - end) laughs per show at after = 3 is at least
-TOLERANCE. TOLERANCE is 1.0 laugh per show of 120 possible levels (0.0083 reward per bit), the bar Exp 18 gave
exploring, and tighter than the -0.02 reward per bit (2.4 laughs per show) it would be if justified by noise alone
(per-bit reward sd is 0.15). The 30-show horizon is reported and does not decide. `after` = 5 (all jokes seen, still decided before the update)
separates the cost of the one-update delay from the cost of seeing only part of the bit.

`audience.yaml` values the result depends on, varied here: `base` gaps between modes (x0.5, x1, x2; judge and human
taste), `noise` (x0.5, x2: how informative three jokes are), `fatigue` (0 and 0.3: how much the last mode matters).
Writes reports/metrics/early_decision.json (cached by params). Importable: run(), table(), criteria(), plot().
"""
import copy
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.director import EPSILON  # noqa: E402
from app.reward import reward_v1  # noqa: E402
from sim.cache import run_or_load  # noqa: E402
from sim.env import AudienceEnv  # noqa: E402
from sim.evaluate_epsilon import EXPLORE_SEED, scale_gaps  # noqa: E402
from sim.evaluate_policies import CONTEXTS, ci  # noqa: E402
from sim.evaluate_tastes import tastes  # noqa: E402
from sim.policies import EpsilonSoft, LinUCB, prior_means  # noqa: E402

AFTERS = (1, 2, 3, 4, 5)  # None (decide at the end of the bit) is the baseline; 5 sees the whole bit but is still one update behind
GATE_AFTER = 3
TOLERANCE = 1.0  # laughs per show
HORIZONS = ((5, 1000), (30, 200))  # (shows, seeds)
GAP_SCALES = (0.5, 1.0, 2.0)
LEVELS_PER_SHOW = 3 * 5 * 8  # laughs per show at reward 1.0, for converting reward per bit


def variants() -> dict:
    out = {f"{taste}, gaps x{s:g}": scale_gaps(a, s) for taste, a in tastes().items() for s in GAP_SCALES}
    judge = tastes()["judge"] if "judge" in tastes() else next(iter(tastes().values()))
    for k in (0.5, 2.0):
        a = copy.deepcopy(judge)
        a["dynamics"]["noise"] = a["dynamics"]["noise"] * k
        out[f"judge, noise x{k:g}"] = a
    for f in (0.0, 0.3):
        a = copy.deepcopy(judge)
        a["dynamics"]["fatigue"] = f
        out[f"judge, fatigue {f:g}"] = a
    return out


def career(audience: dict, shows: int, seed: int, after: int | None) -> float:
    """Mean laughs per show over one career. `after` None decides each bit once the previous one has ended."""
    env = AudienceEnv(audience)
    policy = EpsilonSoft(LinUCB(prior_means(env, "uniform", seed=seed), seed=seed), EPSILON, env.n_modes,
                         seed=seed + EXPLORE_SEED)
    ctx_rng = np.random.default_rng(seed)
    total = 0.0
    for s in range(shows):
        obs = env.reset(CONTEXTS[ctx_rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
        action = policy.decide(obs)[0]
        done = False
        while not done:
            nxt, reward, done, info = env.step(action)
            total += sum(info["levels"])
            if after is not None and not done:
                # decided on the first `after` reactions, before this bit's update
                seen = {**nxt, "last_reward": reward_v1(info["levels"][:after]), "tonight": nxt["tonight"][:-1]}
                chosen = policy.decide(seen)[0]
                policy.update(obs, action, reward)
                obs, action = seen, chosen
            else:
                policy.update(obs, action, reward)
                if not done:
                    obs, action = nxt, policy.decide(nxt)[0]
    return total / shows


def _job(args):
    return career(*args)


def paired(audience: dict, shows: int, seeds: int, pool) -> dict:
    laughs = {}
    for after in (None, *AFTERS):
        laughs[after] = np.array(list(pool.map(_job, [(audience, shows, seed, after) for seed in range(seeds)],
                                               chunksize=20)))
    base = laughs[None]
    out = {"end": {"laughs": ci(base)}}
    for after in AFTERS:
        out[str(after)] = {"laughs": ci(laughs[after]), "diff": ci(laughs[after] - base)}
    return out


def run(horizons: list = [list(h) for h in HORIZONS]) -> dict:
    out = {"variants": {}}
    with ProcessPoolExecutor(4) as pool:  # 4: a 16-process run exhausted memory
        for name, audience in variants().items():
            out["variants"][name] = {str(shows): paired(audience, shows, seeds, pool) for shows, seeds in horizons}
            print(name, flush=True)
    out["criteria"] = criteria(out)
    return out


def criteria(res: dict) -> dict:
    """After 3 jokes passes a variant if its far end of the 95% CI at 5 shows is within the tolerance."""
    rows = {}
    for name, by_horizon in res["variants"].items():
        mean, half = by_horizon["5"][str(GATE_AFTER)]["diff"]
        rows[name] = {"far_end": mean - half, "pass": bool(mean - half >= -TOLERANCE)}
    return {"variants": rows, "all_pass": all(r["pass"] for r in rows.values())}


def table(res: dict) -> str:
    out = ["| Room | Shows | Decide after joke | Laughs per show | Change vs end of bit (paired, 95% CI) | Passes the bar |",
           "|---|---|---|---|---|---|"]
    for name, by_horizon in res["variants"].items():
        for shows, by_after in by_horizon.items():
            for after, r in by_after.items():
                verdict = ""
                if shows == "5" and after == str(GATE_AFTER):
                    c = res["criteria"]["variants"][name]
                    verdict = f"{'yes' if c['pass'] else 'NO'} (far end {c['far_end']:+.2f})"
                diff = "–" if after == "end" else f"{r['diff'][0]:+.2f} ± {r['diff'][1]:.2f}"
                out.append(f"| {name} | {shows} | {after} | {r['laughs'][0]:.2f} ± {r['laughs'][1]:.2f} | {diff} | {verdict} |")
    return "\n".join(out)


def plot(res: dict):
    """Paired change in laughs per show at 5 shows, decide after joke 3, per room, with the tolerance line."""
    import matplotlib.pyplot as plt

    names = list(res["variants"])
    d = [res["variants"][n]["5"][str(GATE_AFTER)]["diff"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(names))
    ax.errorbar([m for m, _ in d], y, xerr=[h for _, h in d], fmt="o", color="#2a78d6", capsize=3)
    ax.axvline(-TOLERANCE, color="#eb6834", ls="--", label=f"tolerance -{TOLERANCE:g} laugh per show")
    ax.axvline(0, color="#8a8880", lw=1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("laughs per show, decide after joke 3 minus decide at bit end (paired, 95% CI, 5 shows)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run_or_load("early_decision", run, horizons=[list(h) for h in HORIZONS])
    print(table(res))
    print(res["criteria"])

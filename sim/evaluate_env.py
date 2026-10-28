"""Sanity metrics for the simulated audience, before any learning policy exists.

Usage:
    python -m sim.evaluate_env [--episodes 1000]

Answers: is there anything to learn (oracle vs best fixed mode vs random), does the best
mode differ by venue, and how many jokes a learner needs to tell the top two modes apart.
Writes reports/metrics/sim_env.json and prints a markdown table for docs/metrics.md.
"""
import argparse
import json
import math
import pathlib

import numpy as np

from sim.env import AGES, ENERGIES, VENUES, AudienceEnv, Context

OUT = pathlib.Path(__file__).resolve().parent.parent / "reports" / "metrics" / "sim_env.json"
Z95 = 1.96


def run(env: AudienceEnv, ctx: Context, policy, episodes: int, seed: int) -> dict:
    levels, regret = [], []
    for ep in range(episodes):
        env.reset(ctx, seed=seed + ep)
        lv = rg = 0.0
        done = False
        while not done:
            _, _, done, info = env.step(policy(env))
            lv += info["level"]
            rg += info["regret"]
        levels.append(lv)
        regret.append(rg)
    return {"laughs": float(np.mean(levels)), "laughs_sd": float(np.std(levels)), "regret": float(np.mean(regret))}


def oracle(env):
    return int(env.expected_rewards().argmax())


def fixed(i):
    return lambda env: i


def jokes_to_separate(gap: float, sd: float) -> int:
    # Two-arm z-test at 95% with equal pulls: each arm needs 2(z·sd/gap)^2, so total doubles that.
    return math.ceil(4 * (Z95 * sd / gap) ** 2) if gap > 0 else -1


def static_gap(env: AudienceEnv, ctx: Context) -> tuple[str, str, float]:
    """Best vs runner-up on the context part only (no novelty, no fatigue)."""
    env.reset(ctx, seed=0)
    env.t = 10**6  # novelty fully decayed
    r = env.expected_rewards()
    o = r.argsort()[::-1]
    return env.modes[o[0]], env.modes[o[1]], float(r[o[0]] - r[o[1]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()
    env = AudienceEnv()
    rng = np.random.default_rng(0)
    rand = lambda e: int(rng.integers(e.n_modes))  # noqa: E731

    rows = []
    for venue in VENUES:
        ctx = Context(venue, "normal", "mixed")
        res = {"oracle": run(env, ctx, oracle, args.episodes, 0),
               "random": run(env, ctx, rand, args.episodes, 0)}
        fixed_res = {m: run(env, ctx, fixed(i), args.episodes, 0) for i, m in enumerate(env.modes)}
        best_fixed = max(fixed_res, key=lambda m: fixed_res[m]["laughs"])
        best, second, gap = static_gap(env, ctx)
        rows.append({"venue": venue, **{k: v["laughs"] for k, v in res.items()},
                     "best_fixed_mode": best_fixed, "best_fixed": fixed_res[best_fixed]["laughs"],
                     "fixed_neurotic": fixed_res["neurotic"]["laughs"],
                     "best_mode": best, "runner_up": second, "gap": gap,
                     "jokes_to_separate": jokes_to_separate(gap, env_reward_sd(env, ctx))})

    # Across all 27 contexts: how often is the best mode something other than the global favorite?
    best_by_ctx = {}
    for v in VENUES:
        for e in ENERGIES:
            for a in AGES:
                best_by_ctx[f"{v}/{e}/{a}"] = static_gap(env, Context(v, e, a))[0]

    out = {"episodes": args.episodes, "n_jokes": env.n_jokes, "venues": rows,
           "best_mode_by_context": best_by_ctx,
           "distinct_best_modes": sorted(set(best_by_ctx.values()))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"Laughs per {env.n_jokes}-joke set (sum of 0-3 levels), mean of {args.episodes} episodes\n")
    print("| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best vs runner-up gap | Jokes to separate |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['venue']} | {r['oracle']:.1f} | {r['best_fixed']:.1f} ({r['best_fixed_mode']}) | "
              f"{r['fixed_neurotic']:.1f} | {r['random']:.1f} | {r['best_mode']} - {r['runner_up']} = "
              f"{r['gap']:.3f} | {r['jokes_to_separate']} |")
    print(f"\nDistinct best modes over 27 contexts: {out['distinct_best_modes']}")


def env_reward_sd(env: AudienceEnv, ctx: Context, n: int = 20000) -> float:
    """Per-joke reward sd for the best mode, by sampling the latent directly."""
    env.reset(ctx, seed=0)
    env.t = 10**6
    mu = env.mu().max()
    lat = mu + np.random.default_rng(0).normal(0, env.noise, n)
    return float(((lat[:, None] > env.thresholds).sum(1) / 3).std())


if __name__ == "__main__":
    main()

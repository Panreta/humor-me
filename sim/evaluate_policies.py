"""PR 2: compare mode-choice policies × priors over a run of shows.

Usage:
    python -m sim.evaluate_policies [--shows 300 --seeds 10]

Each seed is one "career": `--shows` consecutive shows in random contexts (same context sequence
for every policy, so comparisons are paired). Policies keep learning across shows.

Reports, per policy: laughs/show in the first 10 shows (cold start: walking into rooms with
only the prior), laughs/show in the last 50 (experienced), and total laughs lost vs the oracle.
Writes reports/metrics/policies.json and reports/figures/laughs_lost_across_shows.png.
"""
import argparse
import json
import pathlib
import time

import numpy as np

from sim.env import AGES, ENERGIES, VENUES, AudienceEnv, Context
from sim.policies import EpsilonGreedy, Fixed, LinTS, LinUCB, Random, prior_means

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "metrics" / "policies.json"
FIG = ROOT / "reports" / "figures" / "laughs_lost_across_shows.png"
CONTEXTS = [Context(v, e, a) for v in VENUES for e in ENERGIES for a in AGES]
PRIORS = ("humorgen", "uniform", "wrong")
ALGOS = {"ε-greedy": EpsilonGreedy, "LinUCB": LinUCB, "Thompson (LinTS)": LinTS}
COLD, WARM = 10, 50


class Oracle:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return int(self.env.expected_rewards().argmax())

    def update(self, *_):
        pass


def career(make, n_shows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env = AudienceEnv()
    policy = make(env, seed)
    ctx_rng = np.random.default_rng(seed)
    laughs, lost = np.zeros(n_shows), np.zeros(n_shows)
    for s in range(n_shows):
        obs = env.reset(CONTEXTS[ctx_rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
        done = False
        while not done:
            a = policy.act(obs)
            nxt, r, done, info = env.step(a)
            policy.update(obs, a, r)
            obs = nxt
            laughs[s] += sum(info["levels"])
            # regret is in reward units (mean level / 3) per bit; convert to laughs (sum of levels)
            lost[s] += info["regret"] * 3 * env.jokes_per_bit
    return laughs, lost


def makers(env_modes: tuple[str, ...]) -> dict:
    m = {"oracle": lambda env, s: Oracle(env),
         "fixed neurotic": lambda env, s: Fixed(env_modes.index("neurotic")),
         "random": lambda env, s: Random(len(env_modes), seed=s)}
    for prior in PRIORS:
        for name, cls in ALGOS.items():
            m[f"{name} | {prior}"] = (lambda c, p: lambda env, s: c(prior_means(env, p, seed=s), seed=s))(cls, prior)
    return m


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shows", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    results = {}
    for name, make in makers(AudienceEnv().modes).items():
        runs = [career(make, args.shows, seed) for seed in range(args.seeds)]
        laughs = np.stack([r[0] for r in runs])
        lost = np.stack([r[1] for r in runs])
        results[name] = {
            "cold": ci(laughs[:, :COLD].mean(1)),
            "warm": ci(laughs[:, -WARM:].mean(1)),
            "lost_total": ci(lost.sum(1)),
            "lost_curve": [round(v, 1) for v in lost.cumsum(1).mean(0)],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # One line per policy keeps the file diffable (curves are 300 numbers each).
    lines = ",
".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in results.items())
    OUT.write_text(f'{{"shows": {args.shows}, "seeds": {args.seeds}, "results": {{
{lines}
}}}}
')
    plot(results, args.shows)

    print(f"{args.seeds} careers x {args.shows} shows, {time.time() - t0:.0f}s. Laughs/show (40 jokes), mean ± 95% CI\n")
    print(f"| Policy | Prior | First {COLD} shows | Last {WARM} shows | Laughs lost vs oracle (total) |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        algo, _, prior = name.partition(" | ")
        print(f"| {algo} | {prior or '-'} | {r['cold'][0]:.1f} ± {r['cold'][1]:.1f} | "
              f"{r['warm'][0]:.1f} ± {r['warm'][1]:.1f} | {r['lost_total'][0]:.0f} ± {r['lost_total'][1]:.0f} |")


def plot(results: dict, n_shows: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(ALGOS, ("#2a78d6", "#eb6834", "#1baf7a")))  # dataviz categorical slots 1-3
    ink, muted, grid = "#2b2a27", "#8a8880", "#e6e4dd"
    fig, axes = plt.subplots(1, len(PRIORS), figsize=(12, 4), sharey=True)
    x = np.arange(1, n_shows + 1)
    titles = {"humorgen": "HumorGen prior (matches the sim)", "uniform": "Uniform prior",
              "wrong": "Wrong prior (shuffled)"}
    for ax, prior in zip(axes, PRIORS):
        for base, style in (("random", ":"), ("fixed neurotic", "--")):
            ax.plot(x, results[base]["lost_curve"], color=muted, lw=1.5, ls=style, label=base)
        ends = []
        for algo, c in colors.items():
            y = results[f"{algo} | {prior}"]["lost_curve"]
            ax.plot(x, y, color=c, lw=2, label=algo)
            ends.append([y[-1], algo.split(" ")[0]])
        # Dodge end labels so converging lines stay readable.
        gap = 0.05 * max(results["random"]["lost_curve"])
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y_end, label in ends:
            ax.annotate(label, (x[-1], y_end), xytext=(4, 0), textcoords="offset points",
                        color=ink, fontsize=8, va="center")
        ax.set_title(titles[prior], color=ink, fontsize=10, loc="left")
        ax.set_xlabel("shows played", color=ink)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("cumulative laughs lost vs oracle", color=ink)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Laughs lost across a career of shows (lower is better)", color=ink, x=0.01, ha="left")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()

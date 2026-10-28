"""Mid-set preference shift: can the director notice the room turning and switch?

Usage:
    python -m sim.evaluate_shift [--shows 400 --seeds 10]

Each career: `--shows` shows in random rooms; in a random half of them the room's taste shifts
after bit 4 of 8 (audience.yaml `shift`: the modes that were working drop, others rise). The
director is never told. It learns throughout; metrics come from the last 100 shows (experienced).

Compares the chosen director (LinUCB, uniform prior) with and without the reaction feature
(same mode × last reward), plus LinTS and baselines. Writes reports/metrics/shift.json.
"""
import argparse
import json
import pathlib
import time

import numpy as np

from sim.env import AudienceEnv
from sim.evaluate_policies import CONTEXTS, Oracle, ci
from sim.policies import Fixed, LinTS, LinUCB, Random, prior_means

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "metrics" / "shift.json"
SHIFT_AT = 4
EVAL = 100
RECOVERED = 0.02  # per-bit regret (reward units, ≈ 0.3 laughs) counted as "back on the best mode"


def policies(modes) -> dict:
    uni = lambda env, s: prior_means(env, "uniform")  # noqa: E731
    return {
        "oracle": lambda env, s: Oracle(env),
        "fixed neurotic": lambda env, s: Fixed(modes.index("neurotic")),
        "random": lambda env, s: Random(len(modes), seed=s),
        "LinUCB": lambda env, s: LinUCB(uni(env, s), seed=s),
        "LinUCB + react": lambda env, s: LinUCB(uni(env, s), seed=s, react=True),
        "LinTS": lambda env, s: LinTS(uni(env, s), seed=s),
        "LinTS + react": lambda env, s: LinTS(uni(env, s), seed=s, react=True),
    }


def career(make, n_shows: int, seed: int) -> list[dict]:
    plain, shifting = AudienceEnv(), AudienceEnv(shift_at=SHIFT_AT)
    # One policy object across both envs: the director can't tell which room it's in.
    policy = make(plain, seed)
    rng = np.random.default_rng(seed)
    shows = []
    for s in range(n_shows):
        shifted = bool(rng.random() < 0.5)
        env = shifting if shifted else plain
        if isinstance(policy, Oracle):
            policy.env = env
        obs = env.reset(CONTEXTS[rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
        laughs, regret = [], []
        done = False
        while not done:
            a = policy.act(obs)
            nxt, r, done, info = env.step(a)
            policy.update(obs, a, r)
            obs = nxt
            laughs.append(sum(info["levels"]))
            regret.append(info["regret"])
        shows.append({"shifted": shifted, "laughs": laughs, "regret": regret})
    return shows


def bits_to_recover(regret: list[float]) -> int:
    """Bits after the shift until the director first picks a (near-)best mode; censored at 4."""
    for k, r in enumerate(regret[SHIFT_AT:]):
        if r < RECOVERED:
            return k
    return len(regret) - SHIFT_AT


def summarize(careers: list[list[dict]]) -> dict:
    out = {}
    for label, want in (("shifted", True), ("steady", False)):
        pre, post, rec = [], [], []
        for shows in careers:
            sel = [sh for sh in shows[-EVAL:] if sh["shifted"] == want]
            pre.append(np.mean([sum(sh["laughs"][:SHIFT_AT]) for sh in sel]))
            post.append(np.mean([sum(sh["laughs"][SHIFT_AT:]) for sh in sel]))
            rec.append(np.mean([bits_to_recover(sh["regret"]) for sh in sel]))
        out[label] = {"pre": ci(np.array(pre)), "post": ci(np.array(post)), "recover": ci(np.array(rec))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shows", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    results = {name: summarize([career(make, args.shows, s) for s in range(args.seeds)])
               for name, make in policies(AudienceEnv().modes).items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in results.items())
    OUT.write_text(f'{{"shows": {args.shows}, "seeds": {args.seeds}, "shift_at": {SHIFT_AT}, '
                   f'"eval_last": {EVAL}, "results": {{\n{lines}\n}}}}\n')

    o = results["oracle"]
    print(f"{args.seeds} careers x {args.shows} shows ({time.time() - t0:.0f}s); last {EVAL} shows. "
          f"Laughs in bits 5-8 (after the shift point), mean ± 95% CI\n")
    print("| Policy | Shifted shows: after shift | % of oracle | Bits to recover (max 4) | Steady shows: bits 5-8 | % of oracle |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        sp, st = r["shifted"]["post"], r["steady"]["post"]
        print(f"| {name} | {sp[0]:.1f} ± {sp[1]:.1f} | {100 * sp[0] / o['shifted']['post'][0]:.0f}% | "
              f"{r['shifted']['recover'][0]:.2f} ± {r['shifted']['recover'][1]:.2f} | "
              f"{st[0]:.1f} ± {st[1]:.1f} | {100 * st[0] / o['steady']['post'][0]:.0f}% |")


if __name__ == "__main__":
    main()

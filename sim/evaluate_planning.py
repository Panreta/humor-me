"""Is the oracle the true ceiling? It knows the room's truth but picks the best *next* bit, so it can't
plan around fatigue. This compares it with the exact best sequence of modes for each room.

Usage:
    python -m sim.evaluate_planning

Writes reports/metrics/planning_optimum.json (cached by params). Importable: run(), table().
"""
import numpy as np

from sim.cache import run_or_load
from sim.env import MAX_LEVEL, AudienceEnv
from sim.evaluate_policies import COLD, CONTEXTS, Oracle, WARM, career, ci
from sim.evaluate_tastes import CHOSEN, tastes
from sim.policies import LinUCB, prior_means


class PlanningOracle:
    """Knows the truth and follows the best path through the rest of the show, by dynamic programming.

    Only the bit number and the modes inside the fatigue window affect what comes next, so those are the state.
    """

    def __init__(self, env: AudienceEnv):
        self.env = env
        self.window = env.p["dynamics"]["fatigue_window"]
        self._rewards: dict = {}
        self._values: dict = {}

    def _state(self, t: int, recent: tuple) -> tuple:
        return self.env.context, t, recent[-self.window:]

    def rewards(self, t: int, recent: tuple) -> np.ndarray:
        key = self._state(t, recent)
        if key not in self._rewards:
            saved = self.env.t, self.env.history
            self.env.t, self.env.history = t, list(recent)
            self._rewards[key] = self.env.expected_rewards()
            self.env.t, self.env.history = saved
        return self._rewards[key]

    def _action_values(self, t: int, recent: tuple) -> np.ndarray:
        return self.rewards(t, recent) + [self.value(t + 1, recent + (m,)) for m in range(self.env.n_modes)]

    def value(self, t: int, recent: tuple = ()) -> float:
        """Best expected total reward from bit t to the end of the show."""
        if t == self.env.n_bits:
            return 0.0
        recent = recent[-self.window:]
        key = self._state(t, recent)
        if key not in self._values:
            self._values[key] = float(self._action_values(t, recent).max())
        return self._values[key]

    def act(self, obs: dict) -> int:
        recent = tuple(self.env.history[-self.window:])
        return int(self._action_values(obs["t"], recent).argmax())

    def update(self, *_):
        pass


def _play(env: AudienceEnv, pick) -> tuple[float, list[int]]:
    """Expected laughs of one noise-free show, and the modes it used."""
    total = 0.0
    while env.t < env.n_bits:
        expected = env.expected_rewards()
        m = pick(expected)
        total += expected[m]
        env.history.append(m)
        env.t += 1
    return total * MAX_LEVEL * env.jokes_per_bit, list(env.history)


def repeat_rate(modes: list[int]) -> float:
    return float(np.mean([a == b for a, b in zip(modes, modes[1:])]))


def rooms(audience: dict | None = None) -> list[dict]:
    """Expected laughs per show in each room. No sampling, so these numbers carry no noise."""
    env = AudienceEnv(audience)
    out = []
    for c in CONTEXTS:
        env.reset(c, seed=0)
        planner = PlanningOracle(env)
        planned, plan_modes = _play(env, lambda _: planner.act({"t": env.t}))
        env.reset(c, seed=0)
        greedy, greedy_modes = _play(env, lambda e: int(e.argmax()))
        fixed = []
        for m in range(env.n_modes):
            env.reset(c, seed=0)
            fixed.append(_play(env, lambda _: m)[0])
        out.append({"room": f"{c.venue}/{c.energy}/{c.age}", "planning": planned, "oracle": greedy,
                    "best_fixed": max(fixed), "plan": [env.modes[m] for m in plan_modes],
                    "repeats_plan": repeat_rate(plan_modes), "repeats_oracle": repeat_rate(greedy_modes)})
    return out


def makers() -> dict:
    return {"oracle": lambda env, s: Oracle(env), "planning optimum": lambda env, s: PlanningOracle(env),
            CHOSEN: lambda env, s: LinUCB(prior_means(env, "uniform", seed=s), seed=s)}


def run(shows: int = 300, seeds: int = 10) -> dict:
    out = {}
    for taste, audience in tastes().items():
        rows = {}
        for name, make in makers().items():
            runs = [career(make, shows, seed, audience=audience) for seed in range(seeds)]
            laughs, lost = np.stack([r[0] for r in runs]), np.stack([r[1] for r in runs])
            rows[name] = {"cold": ci(laughs[:, :COLD].mean(1)), "warm": ci(laughs[:, -WARM:].mean(1)),
                          "lost_total": ci(lost.sum(1))}
        per_room = rooms(audience)
        planning = np.array([r["planning"] for r in per_room])
        oracle = np.array([r["oracle"] for r in per_room])
        out[taste] = {
            "rows": rows, "rooms": per_room,
            "planning": float(planning.mean()), "oracle": float(oracle.mean()),
            "best_fixed": float(np.mean([r["best_fixed"] for r in per_room])),
            "oracle_share": float((oracle / planning).mean()), "oracle_share_worst": float((oracle / planning).min()),
            "rooms_planning_wins": int((planning > oracle + 1e-9).sum()),
            "repeats_plan": float(np.mean([r["repeats_plan"] for r in per_room])),
            "repeats_oracle": float(np.mean([r["repeats_oracle"] for r in per_room])),
            "chosen_of_oracle": rows[CHOSEN]["warm"][0] / rows["oracle"]["warm"][0],
            "chosen_of_planning": rows[CHOSEN]["warm"][0] / rows["planning optimum"]["warm"][0],
        }
    return out


def table(res: dict) -> str:
    j, h = res["judge"], res["human"]
    rows = [("Expected laughs a show (exact): planning optimum", "planning", ".2f"),
            ("Expected laughs a show (exact): oracle", "oracle", ".2f"),
            ("Expected laughs a show (exact): best fixed mode per room", "best_fixed", ".2f"),
            ("Oracle's share of the optimum, mean of rooms", "oracle_share", ".1%"),
            ("Oracle's share of the optimum, worst room", "oracle_share_worst", ".1%"),
            ("Rooms where planning wins, of 27", "rooms_planning_wins", "d"),
            ("Repeats of the last mode: optimum", "repeats_plan", ".0%"),
            ("Repeats of the last mode: oracle", "repeats_oracle", ".0%"),
            ("LinUCB's share of the oracle, last 50 shows", "chosen_of_oracle", ".1%"),
            ("LinUCB's share of the optimum, last 50 shows", "chosen_of_planning", ".1%")]
    out = ["| | Judge taste | Human taste |", "|---|---|---|"]
    out += [f"| {label} | {j[k]:{f}} | {h[k]:{f}} |" for label, k, f in rows]
    for name in makers():
        a, b = j["rows"][name], h["rows"][name]
        label = name.replace(" | ", ", ")  # a bare pipe would split the markdown row
        out.append(f"| {label}: last {WARM} shows; laughs lost | {a['warm'][0]:.1f} ± {a['warm'][1]:.1f}; {a['lost_total'][0]:.0f} ± {a['lost_total'][1]:.0f} | "
                   f"{b['warm'][0]:.1f} ± {b['warm'][1]:.1f}; {b['lost_total'][0]:.0f} ± {b['lost_total'][1]:.0f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(run_or_load("planning_optimum", run, shows=300, seeds=10)))

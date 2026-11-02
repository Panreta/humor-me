"""The planning optimum must really be the best sequence; short shows only."""
import itertools

import pytest

from sim.env import AudienceEnv, Context, load_audience
from sim.evaluate_planning import PlanningOracle, repeat_rate, rooms

ROOM = Context("bar", "normal", "mixed")


def _env(fatigue=None, n_bits=3):
    audience = load_audience()
    if fatigue is not None:
        audience["dynamics"]["fatigue"] = fatigue
    env = AudienceEnv(audience, n_bits=n_bits)
    env.reset(ROOM, seed=0)
    return env


def _sequence_value(env, modes):
    env.reset(ROOM, seed=0)
    total = 0.0
    for m in modes:
        total += env.expected_rewards()[m]
        env.history.append(m)
        env.t += 1
    env.reset(ROOM, seed=0)
    return total


def test_value_matches_trying_every_sequence():
    env = _env()
    best = max(_sequence_value(env, s) for s in itertools.product(range(env.n_modes), repeat=env.n_bits))
    assert PlanningOracle(env).value(0) == pytest.approx(best)


def test_asking_for_a_value_leaves_the_show_untouched():
    env = _env()
    env.step(0)
    t, history = env.t, list(env.history)
    PlanningOracle(env).value(env.t, tuple(env.history))
    assert (env.t, env.history) == (t, history)


def test_without_fatigue_planning_equals_picking_the_best_next_bit():
    env = _env(fatigue=0.0, n_bits=4)
    greedy = 0.0
    for _ in range(env.n_bits):
        greedy += env.expected_rewards().max()
        env.t += 1
    env.reset(ROOM, seed=0)
    assert PlanningOracle(env).value(0) == pytest.approx(greedy)


def test_playing_the_policy_earns_its_value():
    env = _env(n_bits=4)
    planner = PlanningOracle(env)
    predicted, total, obs = planner.value(0), 0.0, {"t": 0}
    for _ in range(env.n_bits):
        m = planner.act(obs)
        total += env.expected_rewards()[m]
        obs, *_ = env.step(m)
    assert total == pytest.approx(predicted)


def test_rooms_never_score_the_optimum_below_the_oracle_or_a_fixed_mode():
    for r in rooms():
        assert r["planning"] >= r["oracle"] - 1e-9 and r["planning"] >= r["best_fixed"] - 1e-9


def test_repeat_rate():
    assert repeat_rate([1, 1, 2, 2]) == pytest.approx(2 / 3)

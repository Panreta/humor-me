"""Simulated audience env: dynamics behave as documented in sim/priors.yaml."""
import numpy as np
import pytest

from sim.env import AudienceEnv, Context

BAR = Context("bar", "normal", "mixed")


def run(env, actions, seed=0, ctx=BAR):
    env.reset(ctx, seed=seed)
    return [env.step(a) for a in actions]


def test_seeded_episodes_are_reproducible():
    a = [r[1] for r in run(AudienceEnv(), [0, 1, 2] * 5)]
    b = [r[1] for r in run(AudienceEnv(), [0, 1, 2] * 5)]
    assert a == b


def test_reward_is_level_over_three():
    for _, reward, _, info in run(AudienceEnv(), [0] * 40):
        assert info["level"] in (0, 1, 2, 3)
        assert reward == info["level"] / 3


def test_expected_reward_matches_sampling():
    env = AudienceEnv()
    env.reset(BAR, seed=1)
    exp = env.expected_rewards()[1]
    mu = env.mu()[1]
    lat = mu + env.rng.normal(0, env.noise, 200_000)
    mc = (lat[:, None] > env.thresholds).sum(1).mean() / 3
    assert abs(exp - mc) < 0.01


def test_each_venue_has_a_different_best_mode():
    env = AudienceEnv()
    best = set()
    for v in ("bar", "company_event", "showcase"):
        env.reset(Context(v, "normal", "mixed"), seed=0)
        best.add(int(env.expected_rewards().argmax()))
    assert len(best) == 3


def test_repeating_a_mode_tires_the_room_and_recovers():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    for _ in range(10):  # get past the novelty bump
        env.step(2)
    fresh = env.expected_rewards()[1]
    for _ in range(3):
        env.step(1)
    assert env.expected_rewards()[1] < fresh
    for _ in range(5):  # fatigue window passes
        env.step(2)
    assert env.expected_rewards()[1] == pytest.approx(fresh, abs=0.02)


def test_novelty_fades():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    start = env.mu().copy()
    for i in range(20):
        env.step(i % 6)  # spread use so fatigue is equal
    assert (env.mu() < start).all()


def test_mid_set_shift_changes_the_best_mode():
    env = AudienceEnv(shift_at=20)
    env.reset(BAR, seed=0)
    before = int(env.expected_rewards().argmax())
    for i in range(20):
        env.step(i % 6)
    assert int(env.expected_rewards().argmax()) != before


def test_regret_zero_for_oracle_action():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    _, _, _, info = env.step(int(env.expected_rewards().argmax()))
    assert info["regret"] == 0


def test_episode_ends_and_guards():
    env = AudienceEnv(n_jokes=3)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(BAR, seed=0)
    dones = [env.step(0)[2] for _ in range(3)]
    assert dones == [False, False, True]
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(BAR)
    with pytest.raises(ValueError):
        env.step(6)


def test_random_context_when_none_given():
    env = AudienceEnv()
    ctxs = {env.reset(seed=s)["context"] for s in range(30)}
    assert len(ctxs) > 5


def test_unknown_mode_in_priors_fails_loudly():
    with pytest.raises(ValueError):
        AudienceEnv(shift={"clown": 1.0})

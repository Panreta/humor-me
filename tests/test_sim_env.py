"""Simulated audience env: dynamics behave as documented in sim/audience.yaml."""
import pytest

from sim.env import AudienceEnv, Context

BAR = Context("bar", "normal", "mixed")


def run(env, actions, seed=0, ctx=BAR):
    env.reset(ctx, seed=seed)
    return [env.step(a) for a in actions]


def test_seeded_episodes_are_reproducible():
    a = [r[1] for r in run(AudienceEnv(), [0, 1, 2, 3])]
    b = [r[1] for r in run(AudienceEnv(), [0, 1, 2, 3])]
    assert a == b


def test_a_bit_is_several_jokes_and_reward_is_mean_level_over_three():
    env = AudienceEnv()
    for _, reward, _, info in run(env, [0] * env.n_bits):
        assert len(info["levels"]) == env.jokes_per_bit
        assert all(lv in (0, 1, 2, 3) for lv in info["levels"])
        assert reward == pytest.approx(sum(info["levels"]) / env.jokes_per_bit / 3)


def test_expected_reward_matches_sampling():
    env = AudienceEnv()
    env.reset(BAR, seed=1)
    exp = env.expected_rewards()[1]
    total = 0.0
    n = 4000
    for i in range(n):
        env.reset(BAR, seed=1000 + i)
        total += env.step(1)[1]
    assert abs(exp - total / n) < 0.01


def test_each_venue_has_a_different_best_mode():
    env = AudienceEnv()
    best = set()
    for v in ("bar", "company_event", "showcase"):
        env.reset(Context(v, "normal", "mixed"), seed=0)
        best.add(int(env.expected_rewards().argmax()))
    assert len(best) == 3


def test_repeating_a_mode_across_bits_tires_the_room_and_recovers():
    env = AudienceEnv(n_bits=20)
    env.reset(BAR, seed=0)
    for _ in range(4):  # past the novelty bump
        env.step(2)
    fresh = env.expected_rewards()[1]
    env.step(1)
    assert env.expected_rewards()[1] < fresh
    env.step(2)
    env.step(2)  # fatigue window (2 bits) passes
    assert env.expected_rewards()[1] == pytest.approx(fresh, abs=0.01)


def test_novelty_fades_within_and_across_bits():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    assert (env.mu(4) < env.mu(0)).all()
    start = env.mu().copy()
    for i in range(4):
        env.step(i)
    assert (env.mu() < start + 1e-9).all()


def test_mid_set_shift_changes_the_best_mode():
    env = AudienceEnv(shift_at=4)
    env.reset(BAR, seed=0)
    before = int(env.expected_rewards().argmax())
    for i in range(4):
        env.step(i)
    assert int(env.expected_rewards().argmax()) != before


def test_regret_zero_for_oracle_action():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    _, _, _, info = env.step(int(env.expected_rewards().argmax()))
    assert info["regret"] == 0


def test_episode_ends_and_guards():
    env = AudienceEnv(n_bits=3)
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


def test_obs_carries_last_bit_for_history_aware_policies():
    env = AudienceEnv()
    env.reset(BAR, seed=0)
    obs, reward, _, _ = env.step(3)
    assert obs["last_mode"] == 3 and obs["last_reward"] == reward and obs["t"] == 1


def test_random_context_when_none_given():
    env = AudienceEnv()
    ctxs = {env.reset(seed=s)["context"] for s in range(30)}
    assert len(ctxs) > 5


def test_unknown_mode_in_priors_fails_loudly():
    with pytest.raises(ValueError):
        AudienceEnv(shift={"clown": 1.0})

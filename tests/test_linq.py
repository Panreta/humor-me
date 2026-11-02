"""Show state step C1: LinQ's planning inputs and its Bellman target."""
import numpy as np
import pytest

from sim.env import AudienceEnv, Context
from sim.policies import LinQ, LinQRefit, LinUCB, n_features, planning_features, prior_means


def _obs(t, tonight, n_bits=8):
    last = tonight[-1] if tonight else (None, None)
    return {"t": t, "n_bits": n_bits, "context": Context("bar", "normal", "mixed"), "last_mode": last[0],
            "last_reward": last[1], "tonight": tuple(tonight)}


def test_planning_inputs_see_fatigue_a_drop_and_untried_modes():
    obs = _obs(3, [(1, 0.6), (2, 0.6), (3, 0.2)])
    used_two_ago, left, drop, drop_untried, untried_left = planning_features(obs, arm=2)
    assert used_two_ago == 1.0 and left == 5 / 8
    assert abs(drop - 0.4) < 1e-9 and drop_untried == 0.0 and untried_left == 0.0
    _, _, _, drop_untried, untried_left = planning_features(obs, arm=5)
    assert abs(drop_untried - 0.4) < 1e-9 and untried_left == 5 / 8


def test_no_drop_is_counted_when_the_last_bit_did_not_fall():
    assert planning_features(_obs(2, [(1, 0.2), (2, 0.6)]), arm=0)[2] == 0.0
    assert planning_features(_obs(0, []), arm=0)[2] == 0.0


def test_target_adds_the_next_bits_best_value_except_on_the_last_bit():
    q = LinQ(np.full(2, 0.4))
    first, last = _obs(0, [], n_bits=2), _obs(1, [(0, 0.5)], n_bits=2)
    q.update(first, 0, 0.5)
    assert q._pending is not None  # waits for the next bit's observation
    b0, b1 = q.b[0].copy(), q.b[1].copy()
    best_next = q.scores(last)[:, 0].max()
    q.update(last, 1, 0.3)
    assert np.allclose(q.b[0] - b0, (0.5 + best_next) * q._x(first, 0))
    assert np.allclose(q.b[1] - b1, 0.3 * q._x(last, 1))  # reward only: nothing follows the last bit
    assert q._pending is None


def test_last_bit_of_a_show_does_not_leak_into_the_next_show():
    q = LinQ(np.full(2, 0.4))
    q.update(_obs(0, [], n_bits=1), 0, 0.5)
    assert q._pending is None


def test_refit_starts_each_mode_on_the_scale_of_the_show_left():
    q = LinQRefit(np.array([0.4, 0.2]), n_bits=8)
    v = q.scores(_obs(0, []))[:, 0]
    later = q.scores(_obs(6, [(0, 0.5)] * 6))[:, 0]
    assert v[0] > v[1] > later[1]
    assert abs(v[0] / v[1] - 2) < 0.2  # keeps the prior's ratio between modes


def test_refit_bonus_shrinks_as_the_show_runs_out():
    q = LinQRefit(np.array([0.4, 0.2]), n_bits=8)
    obs = _obs(3, [(0, 0.5)] * 3)
    late = dict(obs, t=7)
    assert q.scores(obs)[0, 1] > q.scores(late)[0, 1]


def test_refit_recomputes_old_targets_from_the_current_model():
    q = LinQRefit(np.array([0.4, 0.4]), n_bits=2)
    first, last = _obs(0, [], n_bits=2), _obs(1, [(0, 0.5)], n_bits=2)
    q.update(first, 0, 0.5)
    q.update(last, 1, 0.3)
    after_one_show = q.b[0].copy()
    for _ in range(50):  # a later show teaches that the last bit pays far more
        q.update(first, 1, 0.0)
        q.update(last, 1, 1.0)
    x = q._x(first, 0)
    assert x @ q.b[0] > x @ after_one_show  # mode 0's old bit now targets the higher next value
    assert len(q.shows) == 51 and q._show == []


@pytest.mark.xfail(strict=True, reason="known bug: C1b grows A mid-show while b waits for the show's end (decisions.md)")
def test_an_in_show_update_leaves_values_unchanged_until_the_refit():
    q = LinQRefit(np.array([0.4, 0.4]), n_bits=8)
    obs = _obs(0, [])
    before = q.scores(obs)[0, 0]
    q.update(obs, 0, 0.5)
    assert abs(q.scores(obs)[0, 0] - before) < 1e-9


def test_adding_linq_leaves_linucb_feature_size_alone():
    assert LinUCB(np.full(2, 0.4)).A.shape[1] == n_features()


def test_linq_plays_a_whole_show_and_stays_finite():
    env = AudienceEnv()
    q = LinQ(prior_means(env, "uniform"), seed=0)
    for s in range(3):
        obs, done = env.reset(seed=s), False
        while not done:
            a = q.act(obs)
            nxt, r, done, _ = env.step(a)
            q.update(obs, a, r)
            obs = nxt
    assert np.isfinite(q.b).all() and q._pending is None

"""Experiment functions behind notebooks/decisions.ipynb: cache semantics and tiny runs."""
import pytest

from sim import cache, evaluate_env, evaluate_policies, evaluate_shift


@pytest.fixture
def tmp_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "METRICS", tmp_path)
    return tmp_path


def test_cache_reuses_same_params_and_reruns_on_change(tmp_metrics):
    calls = []

    def fn(x):
        calls.append(x)
        return {"value": x * 2}

    assert cache.run_or_load("t", fn, x=1) == {"value": 2}
    assert cache.run_or_load("t", fn, x=1) == {"value": 2}
    assert calls == [1]                      # second call came from disk
    cache.run_or_load("t", fn, x=2)
    assert calls == [1, 2]                   # new params → recompute
    cache.run_or_load("t", fn, rerun=True, x=2)
    assert calls == [1, 2, 2]                # explicit rerun


def test_env_run_and_table():
    res = evaluate_env.run(episodes=3)
    assert [r["venue"] for r in res["venues"]] == ["bar", "company_event", "showcase"]
    assert "| bar |" in evaluate_env.table(res)


def test_per_joke_config_is_reproducible_via_params():
    env = evaluate_env.make_env(n_bits=40, jokes_per_bit=1, fatigue_window=5)
    assert env.n_jokes == 40 and env.p["dynamics"]["fatigue_window"] == 5
    assert evaluate_env.make_env(8, 5, 2).p["dynamics"]["fatigue_window"] == 2  # no leak between envs


def test_policies_run_table_plot():
    res = evaluate_policies.run(shows=3, seeds=2)
    assert len(res["oracle"]["lost_curve"]) == 3
    assert "LinUCB" in evaluate_policies.table(res)
    import matplotlib
    matplotlib.use("Agg")
    assert len(evaluate_policies.plot(res).axes) == 3


def test_shift_run_and_table():
    res = evaluate_shift.run(shows=4, seeds=2, shift_at=2)
    assert set(res["oracle"]) == {"shifted", "steady"}
    assert "LinUCB + react" in evaluate_shift.table(res)

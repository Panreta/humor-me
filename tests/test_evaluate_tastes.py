"""Two-tastes criteria and base mapping; short careers only."""
import pytest

from sim.env import load_audience
from sim.evaluate_tastes import base_from_win_rates, criteria, run


def _rows(chosen=(50, 1), oracle=(51, 1), fixed=(40, 1), other=(50, 1)):
    w = lambda v: {"warm": list(v)}
    return {"oracle": w(oracle), "fixed neurotic": w(fixed), "fixed cynic": w((fixed[0] - 5, 1)),
            "LinUCB | uniform": w(chosen), "ε-greedy | uniform": w(other), "Thompson (LinTS) | uniform": w(other)}


def test_map_reproduces_the_judge_base_in_the_config():
    judge = {"neurotic": 0.634, "cynic": 0.552, "observer": 0.491, "wordsmith": 0.349, "optimist": 0.411,
             "absurdist": 0.558}
    assert base_from_win_rates(judge) == pytest.approx(load_audience()["base"], abs=0.011)


def test_all_criteria_pass_on_a_clear_winner():
    c = criteria(_rows())
    assert c["p1"] and c["p2"] and c["p3"] and c["best_fixed"] == "fixed neurotic"


def test_p1_fails_under_95_percent_of_oracle():
    assert not criteria(_rows(chosen=(47, 1), oracle=(50, 1)))["p1"]


def test_p2_compares_with_the_best_fixed_mode_and_needs_separate_cis():
    assert not criteria(_rows(chosen=(50, 1), fixed=(48.5, 1)))["p2"]


def test_p3_fails_only_when_another_learner_is_clearly_better():
    assert criteria(_rows(chosen=(50, 1), other=(51, 1)))["p3"]
    c = criteria(_rows(chosen=(50, 1), oracle=(55, 1), other=(53, 1)))
    assert not c["p3"] and "ε-greedy | uniform" in c["beaten_by"]


def test_run_covers_both_tastes_and_changes_only_base():
    res = run(shows=2, seeds=1)
    assert set(res["tastes"]) == {"judge", "human"}
    assert res["base"]["judge"] != res["base"]["human"]
    assert "fixed optimist" in res["tastes"]["human"]["rows"]

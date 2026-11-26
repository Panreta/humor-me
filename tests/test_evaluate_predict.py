"""The Exp 31(b) harness (D-115): the guess changes no reward, the better guess is the one that ships, and the gate
fires exactly at the bar."""
import pytest

from sim import evaluate_early as ee
from sim import evaluate_predict as ep

AUD = ee.variants()[ep.GATE_ROOM]


def test_the_career_with_guesses_has_the_baseline_laughs_exactly():
    for seed in range(3):
        assert ep.career(AUD, 3, seed)["laughs"] == ee.career(AUD, 3, seed, None)


def test_hit_rates_are_rates_and_the_learning_guess_matches_the_greedy_mode_more_often():
    got = [ep.career(AUD, 3, s) for s in range(20)]
    assert all(0 <= g["hit_learn"] <= 1 and 0 <= g["hit_now"] <= 1 for g in got)
    assert all(g["n"] == 3 * 7 for g in got)  # 7 gaps a show of 8 bits
    assert sum(g["greedy_learn"] for g in got) > sum(g["greedy_now"] for g in got)
    assert all(g["hit_learn"] <= g["greedy_learn"] + 0.3 for g in got)  # exploring picks are the only other source of a hit


def test_run_covers_every_room_and_both_guesses():
    res = ep.run(horizons=[[5, 3]])
    assert len(res["variants"]) == 10
    one = res["variants"][ep.GATE_ROOM]["5"]
    assert set(one) == {"learn", "now", "laughs"} and len(one["learn"]["hit"]) == 2
    assert one["learn"]["seconds_saved_per_show"] == pytest.approx(one["learn"]["hit"][0] * ep.PREP_S * ep.GAPS)
    assert "| 5 |" in ep.table(res)


def result(learn, now, lows=None):
    rooms = {name: {"5": {"learn": {"hit": [learn, 0.0]}, "now": {"hit": [now, 0.0]}}} for name in ee.variants()}
    return {"variants": rooms}


def test_the_better_guess_in_the_gate_room_is_the_one_gated():
    assert ep.criteria(result(0.7, 0.4))["shipped"] == "learn"
    assert ep.criteria(result(0.4, 0.7))["shipped"] == "now"


@pytest.mark.parametrize("hit,half,ok", [(0.6, 0.1, True), (0.6, 0.11, False), (0.5, 0.0, True), (0.3, 0.0, False)])
def test_the_gate_is_the_lower_end_of_the_ci_against_the_bar(hit, half, ok):
    res = result(hit, 0.0)
    for r in res["variants"].values():
        r["5"]["learn"]["hit"] = [hit, half]
    assert ep.criteria(res)["all_pass"] is ok and ep.BAR == 0.5

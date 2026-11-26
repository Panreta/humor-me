"""The Exp 31 harness (D-115): the end-of-bit arm is the same career as Exp 18's eps = 0.2, the early arm really
differs, and the gate fires exactly at the tolerance."""
import numpy as np
import pytest

from sim import evaluate_early as ee
from sim.evaluate_epsilon import career_laughs

AUD = ee.variants()["judge, gaps x1"]


def test_the_end_of_bit_arm_is_exactly_the_live_director_career_of_exp_18():
    for seed in range(3):
        assert ee.career(AUD, 3, seed, None) == career_laughs(AUD, 3, seed, 0.2)[0]


def test_the_early_arm_is_deterministic_and_differs_from_the_baseline():
    a = [ee.career(AUD, 4, s, 3) for s in range(8)]
    assert a == [ee.career(AUD, 4, s, 3) for s in range(8)]
    assert a != [ee.career(AUD, 4, s, None) for s in range(8)]


def test_run_covers_every_variant_horizon_and_decision_point():
    res = ee.run(horizons=[[5, 3]])
    assert len(res["variants"]) == 10
    one = next(iter(res["variants"].values()))["5"]
    assert set(one) == {"end", *map(str, ee.AFTERS)}
    assert one["end"]["laughs"][0] > 0 and len(one["3"]["diff"]) == 2
    assert "| 5 | 3 |" in ee.table(res)


def test_variants_change_only_what_they_say():
    base = ee.variants()["judge, gaps x1"]
    assert ee.variants()["judge, noise x2"]["dynamics"]["noise"] == 2 * base["dynamics"]["noise"]
    assert ee.variants()["judge, fatigue 0"]["dynamics"]["fatigue"] == 0.0
    assert ee.variants()["judge, fatigue 0"]["base"] == base["base"]


@pytest.mark.parametrize("diff,half,ok", [(-0.9, 0.1, True), (-0.9, 0.11, False), (0.0, 0.0, True), (-2.0, 0.1, False)])
def test_the_gate_is_the_far_end_of_the_ci_against_the_tolerance(diff, half, ok):
    res = {"variants": {"r": {"5": {"3": {"diff": [diff, half]}}}}}
    assert ee.criteria(res)["variants"]["r"]["pass"] is ok and ee.criteria(res)["all_pass"] is ok
    assert ee.TOLERANCE == 1.0

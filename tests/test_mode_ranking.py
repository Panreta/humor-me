"""Mode-ranking estimators on synthetic data; no rJokes needed."""
import itertools

import numpy as np

from explorer.mode_ranking import length_bins, midranks, reg_effects, spearman, win_rates


def test_midranks_average_ties_within_each_group():
    group = np.array([0, 0, 0, 1, 1])
    x = np.array([5.0, 1.0, 5.0, 2.0, 2.0])
    assert midranks(group, x).tolist() == [2.5, 1.0, 2.5, 1.5, 1.5]


def test_win_rates_match_brute_force_pair_counting():
    rng = np.random.default_rng(0)
    mode, cell = rng.integers(0, 3, 300), rng.integers(0, 4, 300)
    pct = rng.integers(0, 5, 300) / 4  # coarse, so plenty of ties
    wins, pairs = np.zeros(3), np.zeros(3)
    for i, j in itertools.permutations(range(300), 2):
        if cell[i] == cell[j] and mode[i] != mode[j]:
            pairs[mode[i]] += 1
            wins[mode[i]] += 1.0 if pct[i] > pct[j] else 0.5 if pct[i] == pct[j] else 0.0
    assert np.allclose(win_rates(mode, cell, pct, n_modes=3), wins / pairs)


def test_win_rates_compare_within_cells_only():
    """Mode 0 lives in a high-scoring cell; inside each cell it's no better than mode 1."""
    mode = np.array([0, 0, 1, 1, 0, 1])
    cell = np.array([0, 0, 0, 0, 1, 1])
    pct = np.array([0.9, 0.8, 0.9, 0.8, 0.1, 0.1])
    assert np.allclose(win_rates(mode, cell, pct, n_modes=2), [0.5, 0.5])


def test_regression_recovers_planted_mode_effects_despite_length_confound():
    rng = np.random.default_rng(1)
    n = 20000
    mode = rng.integers(0, 3, n)
    lbin = np.clip(mode + rng.integers(-1, 2, n), 0, 3)  # mode 2 jokes run long
    year = rng.integers(0, 5, n)
    truth = np.array([0.02, 0.0, -0.02])
    pct = 0.5 + truth[mode] + 0.05 * lbin + rng.normal(0, 0.05, n)
    assert np.allclose(reg_effects(mode, year, lbin, pct, n_modes=3), truth, atol=0.003)


def test_length_bins_are_roughly_equal_sized():
    words = np.arange(1, 1001)
    assert np.bincount(length_bins(words, 4)).tolist() == [250, 250, 250, 250]


def test_spearman_is_rank_only():
    assert spearman(np.array([1.0, 2, 3]), np.array([10.0, 200, 3000])) == 1.0
    assert spearman(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == -1.0


def test_an_extra_adjustment_removes_a_gap_that_only_comes_through_that_feature():
    rng = np.random.default_rng(0)
    n = 40000
    mode = rng.integers(0, 2, n)
    extra = (rng.random(n) < np.where(mode == 1, 0.6, 0.1)).astype(int)
    pct = 0.5 - 0.2 * extra + rng.normal(0, 0.05, n)
    year, lbin = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    without = reg_effects(mode, year, lbin, pct, n_modes=2)
    with_extra = reg_effects(mode, year, lbin, pct, n_modes=2, extra=extra)
    assert abs(without[1] - without[0]) > 0.05
    assert abs(with_extra[1] - with_extra[0]) < 0.005
    cell = extra
    assert abs(win_rates(mode, cell, pct, n_modes=2)[1] - 0.5) < 0.02

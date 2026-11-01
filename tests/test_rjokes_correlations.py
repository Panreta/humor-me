"""Correlation-map helpers on synthetic data; no rJokes needed."""
import numpy as np

from explorer.rjokes_correlations import apply_rule, corr_matrix, nullable, rank_columns, within_cells


def test_rank_correlation_ignores_skew():
    x = np.arange(1.0, 51.0)
    R = rank_columns(np.column_stack([x, np.exp(x / 2)]))
    assert abs(corr_matrix(R)[0, 1] - 1) < 1e-12


def test_a_link_that_only_comes_through_the_cell_fades_within_cells():
    rng = np.random.default_rng(0)
    cell = rng.integers(0, 5, 20000)
    x, y = cell + rng.normal(size=20000), cell + rng.normal(size=20000)
    R = rank_columns(np.column_stack([x, y]))
    assert corr_matrix(R)[0, 1] > 0.5
    assert abs(corr_matrix(within_cells(R, cell))[0, 1]) < 0.03


def test_a_real_link_survives_within_cells():
    rng = np.random.default_rng(0)
    cell = rng.integers(0, 5, 20000)
    x = rng.normal(size=20000)
    y = x + rng.normal(size=20000)
    R = rank_columns(np.column_stack([x, y]))
    assert corr_matrix(within_cells(R, cell))[0, 1] > 0.5


def test_a_column_constant_within_cells_has_no_correlation():
    cell = np.array([0, 0, 1, 1])
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 9.0], [4.0, 9.0]])
    m = corr_matrix(within_cells(rank_columns(X), cell))
    assert np.isnan(m[0, 1]) and nullable(m)[0][1] is None


def _adj(names, links):
    m = [[0.0] * len(names) for _ in names]
    for a, b, v in links:
        m[names.index(a)][names.index(b)] = m[names.index(b)][names.index(a)] = v
    return m


def test_rule_needs_both_links_and_sends_safety_labels_to_a_closer_look():
    names = ["pct", "mode_a", "mode_b", "flag_both", "flag_pct_only", "flag_mode_only", "flag_just_under", "safety_both"]
    adj = _adj(names, [("flag_both", "pct", 0.06), ("flag_both", "mode_b", -0.05),
                       ("flag_pct_only", "pct", 0.2),
                       ("flag_mode_only", "mode_a", 0.3),
                       ("flag_just_under", "pct", 0.049), ("flag_just_under", "mode_a", 0.3),
                       ("safety_both", "pct", -0.1), ("safety_both", "mode_a", 0.1)])
    rule = apply_rule(names, adj)
    assert rule["rerun_for"] == ["flag_both"]
    assert rule["look_closer"] == ["safety_both"]

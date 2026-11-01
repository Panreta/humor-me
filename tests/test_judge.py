"""judge/: Bradley-Terry fit, pair selection, position randomization, and the rank() loop.
Synthetic jokes only; no network."""
import numpy as np

from judge.bt import borda_scores, fit_bt


def test_fit_bt_orders_by_wins_and_centers():
    theta, raw = fit_bt(3, [(0, 1), (0, 2), (1, 2), (0, 1)])
    assert theta[0] > theta[1] > theta[2]
    assert abs(theta.mean()) < 1e-12
    assert np.allclose(theta, raw - raw.mean())


def test_winless_item_gets_finite_theta():
    theta, _ = fit_bt(3, [(0, 2), (1, 2), (0, 2)])
    assert np.all(np.isfinite(theta))
    assert theta[2] == theta.min()


def test_borda_scores():
    b = borda_scores(4, [(0, 1), (0, 1), (1, 0), (0, 2)])
    assert np.isclose(b[0], (2 / 3 + 1) / 2) and b[2] == 0.0 and np.isnan(b[3])

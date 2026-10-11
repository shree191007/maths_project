import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from climatematrix import interpolation as ip


def test_linear_matches_numpy():
    x = np.array([0, 1, 3, 4, 8.0]); y = np.array([1, 0, 2, 5, 3.0]); q = np.linspace(0, 8, 33)
    assert np.allclose(ip.linear_interpolate(x, y, q), np.interp(q, x, y))


def test_global_lagrange_reproduces_polynomial_exactly():
    x = np.linspace(0, 10, 6)
    f = lambda t: 0.5 * t**3 - 2 * t**2 + t + 4
    q = np.linspace(0, 10, 21)
    assert np.allclose(ip.lagrange_interpolate(x, f(x), q), f(q))


def test_local_lagrange_window2_is_linear_and_window4_exact_for_cubics():
    x = np.arange(0, 30.0); y = x**3 / 100
    q = np.array([3.5, 10.2, 20.7])
    assert np.allclose(ip.lagrange_interpolate(x, y, q, window=2), ip.linear_interpolate(x, y, q))
    assert np.allclose(ip.lagrange_interpolate(x, y, q, window=4), q**3 / 100)


def test_lagrange_hits_nodes():
    x = np.array([1.0, 2, 4, 7]); y = np.array([3.0, -1, 2, 5])
    assert np.allclose(ip.lagrange_interpolate(x, y, x), y)


def test_vandermonde_agrees_with_lagrange_for_few_nodes():
    x = np.linspace(1900, 2000, 6); y = np.array([0.1, -0.2, 0.0, 0.3, 0.5, 0.4]); q = np.linspace(1900, 2000, 11)
    assert np.allclose(ip.vandermonde_interpolate(x, y, q), ip.lagrange_interpolate(x, y, q))


def test_vandermonde_condition_grows_fast():
    c = [ip.vandermonde_condition(np.arange(n, dtype=float)) for n in (5, 10, 20)]
    assert c[0] < c[1] < c[2] and c[2] > 1e7


def test_natural_spline_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.sort(rng.choice(np.arange(1880, 2020), 25, replace=False)).astype(float); y = rng.normal(size=25)
    q = np.linspace(x[0], x[-1], 200)
    assert np.allclose(ip.cubic_spline_interpolate(x, y, q), CubicSpline(x, y, bc_type="natural")(q))


def test_spline_has_zero_end_curvature():
    x = np.arange(8.0); y = np.sin(x)
    M = ip.natural_spline_second_derivatives(x, y)
    assert M[0] == 0 and M[-1] == 0


def test_no_extrapolation_and_input_validation():
    with pytest.raises(ValueError):
        ip.linear_interpolate([0, 1, 2], [0, 1, 2], [3])
    with pytest.raises(ValueError):
        ip.linear_interpolate([0, 2, 1], [0, 1, 2], [1])


def test_hidden_mask_properties():
    rng = np.random.default_rng(0)
    for g in (1, 3, 5):
        m = ip.make_hidden_mask(100, 20, g, rng)
        assert not m[0] and not m[-1] and m.sum() > 0
        # runs have length g, separated by known points
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])))[::2]
        assert set(runs) == {g}


def test_holdout_experiment_runs_and_linear_exact_on_line():
    x = np.arange(60.0); y = 0.02 * x + 1
    res = ip.holdout_experiment(x, y, n_trials=5, gap_length=3)
    assert res[res.method == "linear"]["rmse"].max() < 1e-12


def test_runge_experiment_skips_node_counts_larger_than_series():
    x = np.arange(1990.0, 2026); y = np.sin(x / 5)  # 36 points, default counts go up to 40
    df = ip.runge_experiment(x, y)
    assert len(df) > 0 and df["nodes"].max() < len(x)

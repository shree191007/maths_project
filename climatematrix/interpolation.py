"""Part 1: polynomial interpolation of missing annual observations.

Every method answers: given known (year, anomaly) pairs, what is the anomaly
at a year we did not observe?

* linear           - piecewise degree-1 polynomial through neighbouring points.
* lagrange         - P(t) = sum_i T_i L_i(t). With ``window=None`` one polynomial
                     goes through *all* nodes (degree n-1); with ``window=k`` only
                     the k nearest nodes are used (local Lagrange).
* cubic spline     - natural cubic spline; second derivatives come from a
                     tridiagonal linear system.
* vandermonde      - the same global polynomial found by solving V c = y, kept to
                     show how ill-conditioned the matrix form is.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

from . import metrics


def _check(x_known, y_known, x_query=None):
    x = np.asarray(x_known, dtype=float)
    y = np.asarray(y_known, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x_known and y_known must be 1-D arrays of equal length")
    if len(x) < 2:
        raise ValueError("need at least two known points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x_known must be strictly increasing")
    if x_query is None:
        return x, y, None
    q = np.atleast_1d(np.asarray(x_query, dtype=float))
    if q.min() < x[0] or q.max() > x[-1]:
        raise ValueError("query outside the known range: interpolation does not extrapolate")
    return x, y, q


# --------------------------------------------------------------------- linear
def linear_interpolate(x_known, y_known, x_query):
    """Piecewise-linear interpolation: y0 + (y1-y0) (t-x0)/(x1-x0)."""
    x, y, q = _check(x_known, y_known, x_query)
    i = np.clip(np.searchsorted(x, q, side="right"), 1, len(x) - 1)
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return y0 + (y1 - y0) * (q - x0) / (x1 - x0)


# ------------------------------------------------------------------- lagrange
def _lagrange_eval(xn, yn, q):
    """Evaluate sum_i y_i L_i(q), L_i(q) = prod_{j!=i} (q-x_j)/(x_i-x_j)."""
    out = np.zeros_like(q)
    for i in range(len(xn)):
        li = np.ones_like(q)
        for j in range(len(xn)):
            if j != i:
                li *= (q - xn[j]) / (xn[i] - xn[j])
        out += yn[i] * li
    return out


def lagrange_interpolate(x_known, y_known, x_query, window: int | None = None):
    """Lagrange interpolation, global (window=None) or through the ``window``
    nearest known points around each query (window=2 reproduces linear)."""
    x, y, q = _check(x_known, y_known, x_query)
    # Map years to [-1, 1] so the products stay in a sane floating-point range.
    mid, half = (x[0] + x[-1]) / 2, (x[-1] - x[0]) / 2
    xs, qs = (x - mid) / half, (q - mid) / half
    if window is None or window >= len(x):
        return _lagrange_eval(xs, y, qs)
    out = np.empty_like(qs)
    for k, qk in enumerate(qs):
        # nearest `window` nodes, forced to bracket the query when possible
        right = int(np.searchsorted(xs, qk))
        lo = int(np.clip(right - window // 2, 0, len(xs) - window))
        out[k] = _lagrange_eval(xs[lo:lo + window], y[lo:lo + window], np.array([qk]))[0]
    return out


def vandermonde_matrix(x, scale: bool = True) -> np.ndarray:
    """V[i, j] = x_i ** j. With scale=True the nodes are first mapped to [-1, 1]."""
    x = np.asarray(x, dtype=float)
    if scale:
        x = (x - (x[0] + x[-1]) / 2) / ((x[-1] - x[0]) / 2)
    return np.vander(x, increasing=True)


def vandermonde_interpolate(x_known, y_known, x_query):
    """Solve V c = y for monomial coefficients, then evaluate. Same polynomial
    as global Lagrange in exact arithmetic; numerically much worse."""
    x, y, q = _check(x_known, y_known, x_query)
    mid, half = (x[0] + x[-1]) / 2, (x[-1] - x[0]) / 2
    c = np.linalg.solve(vandermonde_matrix(x), y)
    return np.polynomial.polynomial.polyval((q - mid) / half, c)


def vandermonde_condition(x) -> float:
    return float(np.linalg.cond(vandermonde_matrix(x)))


# --------------------------------------------------------------- cubic spline
def natural_spline_second_derivatives(x, y) -> np.ndarray:
    """Solve the tridiagonal system for the spline's second derivatives M_i.

        h_{i-1} M_{i-1} + 2 (h_{i-1}+h_i) M_i + h_i M_{i+1}
              = 6 [ (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} ],   M_0 = M_n = 0
    """
    n = len(x)
    if n < 3:
        return np.zeros(n)
    h = np.diff(x)
    rhs = 6 * (np.diff(y)[1:] / h[1:] - np.diff(y)[:-1] / h[:-1])
    ab = np.zeros((3, n - 2))
    ab[0, 1:] = h[1:-1]                 # super-diagonal
    ab[1, :] = 2 * (h[:-1] + h[1:])     # diagonal
    ab[2, :-1] = h[1:-1]                # sub-diagonal
    M = np.zeros(n)
    M[1:-1] = solve_banded((1, 1), ab, rhs)
    return M


def cubic_spline_interpolate(x_known, y_known, x_query):
    """Natural cubic spline (C^2, zero curvature at both ends)."""
    x, y, q = _check(x_known, y_known, x_query)
    if len(x) < 3:
        return linear_interpolate(x, y, q)
    M = natural_spline_second_derivatives(x, y)
    i = np.clip(np.searchsorted(x, q, side="right"), 1, len(x) - 1)
    x0, x1, y0, y1, M0, M1 = x[i - 1], x[i], y[i - 1], y[i], M[i - 1], M[i]
    h = x1 - x0
    a, b = x1 - q, q - x0
    return (M0 * a**3 + M1 * b**3) / (6 * h) + (y0 / h - M0 * h / 6) * a + (y1 / h - M1 * h / 6) * b


# --------------------------------------------------------------------- registry
METHODS = {
    "linear": linear_interpolate,
    "lagrange (4-pt local)": partial(lagrange_interpolate, window=4),
    "lagrange (6-pt local)": partial(lagrange_interpolate, window=6),
    "cubic spline": cubic_spline_interpolate,
}


# ----------------------------------------------------------- hold-out testing
def make_hidden_mask(n: int, n_hidden: int, gap_length: int, rng: np.random.Generator) -> np.ndarray:
    """Hide ~n_hidden points as runs of ``gap_length`` consecutive points.

    The first and last points are never hidden (interpolation cannot
    extrapolate) and runs are separated by at least one known point.
    """
    mask = np.zeros(n, dtype=bool)
    n_gaps = max(1, round(n_hidden / gap_length))
    placed, attempts = 0, 0
    while placed < n_gaps and attempts < 2000:
        attempts += 1
        start = int(rng.integers(1, n - gap_length))  # leaves index n-1 known
        lo, hi = max(start - 1, 0), min(start + gap_length + 1, n)
        if not mask[lo:hi].any():
            mask[start:start + gap_length] = True
            placed += 1
    return mask


def predict_hidden(years, values, hidden, methods=METHODS) -> dict:
    """Reconstruct the hidden entries with every method; returns {name: preds}."""
    years, values = np.asarray(years, float), np.asarray(values, float)
    known = ~hidden
    return {name: f(years[known], values[known], years[hidden]) for name, f in methods.items()}


def holdout_experiment(years, values, methods=METHODS, frac: float = 0.2, gap_length: int = 1,
                       n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """Repeatedly hide a fraction of the series and score every method.

    Returns one row per (trial, method) with the rmse / mae on the hidden points.
    """
    years, values = np.asarray(years, float), np.asarray(values, float)
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        hidden = make_hidden_mask(len(years), round(frac * len(years)), gap_length, rng)
        for name, pred in predict_hidden(years, values, hidden, methods).items():
            rows.append({"trial": trial, "method": name, "gap_length": gap_length,
                         "rmse": metrics.rmse(values[hidden], pred), "mae": metrics.mae(values[hidden], pred)})
    return pd.DataFrame(rows)


def summarize_holdout(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of per-trial RMSE / MAE for each method (sorted best first)."""
    g = results.groupby("method")[["rmse", "mae"]]
    out = g.mean().join(g.std(), rsuffix="_std")
    return out.sort_values("rmse")


def runge_experiment(years, values, node_counts=(5, 8, 12, 16, 20, 30, 40)) -> pd.DataFrame:
    """Global Lagrange vs. spline as the number of equispaced nodes grows.

    Nodes are equispaced years; the error is measured on all other years.
    Shows the Runge phenomenon and the Vandermonde condition number.
    """
    years, values = np.asarray(years, float), np.asarray(values, float)
    rows = []
    for n in node_counts:
        idx = np.unique(np.round(np.linspace(0, len(years) - 1, n)).astype(int))
        test = np.setdiff1d(np.arange(len(years)), idx)
        if len(test) == 0:  # series too short: every year would be a node
            continue
        xk, yk, xt, yt = years[idx], values[idx], years[test], values[test]
        rows.append({
            "nodes": len(idx),
            "cond_vandermonde": vandermonde_condition(xk),
            "rmse_lagrange_global": metrics.rmse(yt, lagrange_interpolate(xk, yk, xt)),
            "rmse_linear": metrics.rmse(yt, linear_interpolate(xk, yk, xt)),
            "rmse_spline": metrics.rmse(yt, cubic_spline_interpolate(xk, yk, xt)),
        })
    return pd.DataFrame(rows)

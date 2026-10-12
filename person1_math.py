"""Person 1: MATHEMATICS  (interpolation + SVD)

What this file does
    PART A - Interpolation: estimate a missing year from the known years around it.
        * linear_interpolate        straight line between the two neighbouring years
        * lagrange_interpolate      polynomial through several nearby years, P(t) = sum T_i L_i(t)
        * cubic_spline_interpolate  smooth curve; solves a tridiagonal system of equations
        * vandermonde_interpolate   the same polynomial found by solving V c = y (shows why one big
                                    polynomial through every point is numerically hopeless)
        * make_hidden_mask / predict_hidden   hide known years, then rebuild them
    PART B - SVD: turn the climate data into the matrix A (regions x years) and break it into patterns.
        * compute_svd               A = U S V^T  using np.linalg.svd()
        * explained_variance        how much each pattern matters
        * reconstruct               rebuild A from only the first k patterns
        * rank_table                reconstruction error versus rank k (with the Eckart-Young check)
        * impute_lowrank / impute_soft   fill in missing matrix entries from the other regions

Run it
    python person1_math.py              # writes reports/figures/interpolation_*.png, svd_*.png and tables
    python -m pytest person1_math.py    # tests for this file

Needs: person2_data.py (data loading and chart style).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

from person2_data import (BLUE, CAT, DIV, GRAY, INK, INK_2, ORANGE, load_global, new_fig, region_matrix,
                          save_figure, save_table)


def _rmse(y_true, y_pred) -> float:
    """Local copy of the RMSE formula, so this file does not depend on Person 4's file."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ----------------------------------------------------------------------------
# PART A - Interpolation
# ----------------------------------------------------------------------------


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


def linear_interpolate(x_known, y_known, x_query):
    """Piecewise-linear interpolation: y0 + (y1-y0) (t-x0)/(x1-x0)."""
    x, y, q = _check(x_known, y_known, x_query)
    i = np.clip(np.searchsorted(x, q, side="right"), 1, len(x) - 1)
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return y0 + (y1 - y0) * (q - x0) / (x1 - x0)


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


METHODS = {
    "linear": linear_interpolate,
    "lagrange (4-pt local)": partial(lagrange_interpolate, window=4),
    "lagrange (6-pt local)": partial(lagrange_interpolate, window=6),
    "cubic spline": cubic_spline_interpolate,
}


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
            "rmse_lagrange_global": _rmse(yt, lagrange_interpolate(xk, yk, xt)),
            "rmse_linear": _rmse(yt, linear_interpolate(xk, yk, xt)),
            "rmse_spline": _rmse(yt, cubic_spline_interpolate(xk, yk, xt)),
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# PART B - SVD of the region x year matrix
# ----------------------------------------------------------------------------


CENTER_MODES = ("row", "col", "none")


@dataclass
class SVDResult:
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    mean: np.ndarray        # what was subtracted (broadcastable to A)
    weights: np.ndarray     # row weights applied before the SVD (ones if unweighted)

    @property
    def rank(self) -> int:
        return len(self.s)


def center(A: np.ndarray, mode: str = "row"):
    """Subtract row means ('row': each region's own time-mean), column means
    ('col': the cross-region mean of each year) or nothing."""
    if mode not in CENTER_MODES:
        raise ValueError(f"mode must be one of {CENTER_MODES}")
    if mode == "row":
        mu = A.mean(axis=1, keepdims=True)
    elif mode == "col":
        mu = A.mean(axis=0, keepdims=True)
    else:
        mu = np.zeros((1, 1))
    return A - mu, mu


def compute_svd(A: np.ndarray, center_mode: str = "row", weights: np.ndarray | None = None) -> SVDResult:
    """Thin SVD of the centred (and optionally row-weighted) matrix.

    ``weights`` are per-row multipliers, e.g. sqrt(area fraction) so polar
    bands, which cover little area, do not dominate.
    """
    if np.isnan(A).any():
        raise ValueError("A contains NaN: use impute_lowrank for incomplete matrices")
    Ac, mu = center(A, center_mode)
    w = np.ones(A.shape[0]) if weights is None else np.asarray(weights, float)
    U, s, Vt = np.linalg.svd(Ac * w[:, None], full_matrices=False)
    return SVDResult(U, s, Vt, mu, w)


def explained_variance(s: np.ndarray) -> np.ndarray:
    """Fraction of total squared Frobenius norm carried by each component."""
    return s**2 / np.sum(s**2)


def reconstruct(res: SVDResult, k: int) -> np.ndarray:
    """Rank-k approximation of the original (un-centred, un-weighted) matrix."""
    Ak = (res.U[:, :k] * res.s[:k]) @ res.Vt[:k]
    return Ak / res.weights[:, None] + res.mean


def spatial_pattern(res: SVDResult, i: int) -> np.ndarray:
    """Mode i over regions, in physical units (undoing the row weights),
    scaled so its largest magnitude is 1 and its sign is fixed (sum >= 0)."""
    u = res.U[:, i] / res.weights
    u = u / np.max(np.abs(u))
    return u if u.sum() >= 0 else -u


def temporal_amplitude(res: SVDResult, i: int) -> np.ndarray:
    """Time series paired with ``spatial_pattern(res, i)`` (same sign convention)."""
    u = res.U[:, i] / res.weights
    sign = 1.0 if (u / np.max(np.abs(u))).sum() >= 0 else -1.0
    return sign * res.s[i] * res.Vt[i] * np.max(np.abs(u))


def rank_table(A: np.ndarray, res: SVDResult, ks=None) -> pd.DataFrame:
    """Reconstruction error versus retained rank, with the Eckart-Young check.

    ``frob_error`` is measured on the original matrix; ``eckart_young`` is the
    prediction sqrt(sum_{i>k} s_i^2) which equals it exactly for an unweighted fit.
    """
    ks = range(0, res.rank + 1) if ks is None else ks
    ev = explained_variance(res.s)
    total = np.linalg.norm(A - res.mean)
    rows = []
    for k in ks:
        Ak = reconstruct(res, k) if k > 0 else np.broadcast_to(res.mean, A.shape).copy()
        err = A - Ak
        rows.append({"k": k, "explained_cum": float(ev[:k].sum()),
                     "frob_error": float(np.linalg.norm(err)),
                     "eckart_young": float(np.sqrt(np.sum(res.s[k:] ** 2))),
                     "rel_error": float(np.linalg.norm(err) / total),
                     "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows).set_index("k")


def numerical_rank(s: np.ndarray, tol: float | None = None) -> int:
    """Number of singular values above tol (default: eps * max(shape) * s_max)."""
    tol = np.finfo(float).eps * s.max() * len(s) if tol is None else tol
    return int(np.sum(s > tol))


def random_mask(shape, frac: float, rng: np.random.Generator) -> np.ndarray:
    """True where entries are hidden; first/last column kept for fair comparison
    with interpolation (which cannot extrapolate)."""
    m = rng.random(shape) < frac
    m[:, [0, -1]] = False
    return m


def block_mask(shape, gap_length: int, n_gaps_per_region: int, rng: np.random.Generator) -> np.ndarray:
    """Each region loses ``n_gaps_per_region`` runs of ``gap_length`` consecutive years."""
    m = np.zeros(shape, dtype=bool)
    for r in range(shape[0]):
        placed, attempts = 0, 0
        while placed < n_gaps_per_region and attempts < 1000:
            attempts += 1
            start = int(rng.integers(1, shape[1] - gap_length))
            lo, hi = max(start - 1, 0), min(start + gap_length + 1, shape[1])
            if not m[r, lo:hi].any():
                m[r, start:start + gap_length] = True
                placed += 1
    return m


def _complete(A_obs: np.ndarray, truncate, max_iter: int, tol: float):
    """Shared loop: fill gaps, re-centre by row, low-rank-ify with ``truncate``,
    write the result into the gaps only, repeat until the update is below tol."""
    obs = ~np.isnan(A_obs)
    if not obs.any(axis=1).all():
        raise ValueError("every region needs at least one observation")
    X = np.where(obs, A_obs, np.nanmean(A_obs, axis=1, keepdims=True))
    history = []
    for _ in range(max_iter):
        mu = X.mean(axis=1, keepdims=True)
        X_new = np.where(obs, A_obs, truncate(X - mu) + mu)
        history.append(np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-12))
        X = X_new
        if history[-1] < tol:
            break
    return X, history


def impute_lowrank(A_obs: np.ndarray, k: int, max_iter: int = 500, tol: float = 1e-8, return_history: bool = False):
    """Fill NaNs with a rank-k + row-mean model (iterative hard-impute).

    Each pass rebuilds the matrix from its rank-k SVD and replaces *only the
    missing entries*. k=0 is the plain row-mean fill (baseline). With only a
    handful of regions, k >= 3 overfits badly: a year with several gaps leaves
    fewer observed values than free parameters.
    """
    def truncate(Xc):
        if k == 0:
            return np.zeros_like(Xc)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :k] * s[:k]) @ Vt[:k]

    X, history = _complete(A_obs, truncate, max_iter, tol)
    return (X, history) if return_history else X


def impute_soft(A_obs: np.ndarray, lam: float, max_iter: int = 500, tol: float = 1e-8, return_history: bool = False):
    """Soft-threshold completion: shrink every singular value s -> max(s - lam, 0).

    A smooth alternative to choosing a hard rank. ``lam`` is in the units of
    the singular values (deg C times sqrt of matrix size); lam=0 keeps all
    components and overfits just like full rank.
    """
    def truncate(Xc):
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U * np.maximum(s - lam, 0.0)) @ Vt

    X, history = _complete(A_obs, truncate, max_iter, tol)
    return (X, history) if return_history else X

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------


METHOD_COLOR = {"linear": CAT[0], "cubic spline": CAT[1], "lagrange (4-pt local)": CAT[2],
                "lagrange (6-pt local)": CAT[3], "lagrange (global)": CAT[7]}


def plot_reconstruction(years, values, hidden, preds: dict, curves: tuple = ("linear", "cubic spline")):
    """Top: series, hidden truth and interpolant curves. Bottom: error at each hidden year."""
    years, values = np.asarray(years, float), np.asarray(values, float)
    fig, (a1, a2) = new_fig(nrows=2, h=3.2, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    known = ~hidden
    a1.plot(years[known], values[known], "o", color=GRAY, ms=3.5, label="known")
    a1.plot(years[hidden], values[hidden], "o", mfc="none", mec=INK, mew=1.4, ms=7, label="hidden (truth)")
    grid = np.linspace(years[known][0], years[known][-1], 800)
    for name in curves:
        if name in METHODS:
            a1.plot(grid, METHODS[name](years[known], values[known], grid), lw=1.4, color=METHOD_COLOR[name], label=name)
    a1.set_ylabel("anomaly (°C)")
    a1.legend(ncol=4, loc="upper left")
    a1.set_title("Reconstructing hidden years from the remaining observations")
    for name, p in preds.items():
        a2.plot(years[hidden], p - values[hidden], "o", color=METHOD_COLOR.get(name, GRAY), ms=4.5, label=name, alpha=.9)
    a2.axhline(0, color=INK_2, lw=0.8)
    a2.set_ylabel("error (°C)")
    a2.set_xlabel("year")
    lo, hi = a2.get_ylim()
    a2.set_ylim(lo, hi + 0.45 * (hi - lo))
    a2.legend(ncol=4, loc="upper left", fontsize=8)
    return fig


def plot_runge(df):
    """Error and Vandermonde condition number against the number of nodes."""
    fig, (a1, a2) = new_fig(ncols=2, w=5.6, h=3.6)
    for col, label, c in (("rmse_lagrange_global", "global Lagrange", METHOD_COLOR["lagrange (global)"]),
                          ("rmse_linear", "linear", METHOD_COLOR["linear"]),
                          ("rmse_spline", "cubic spline", METHOD_COLOR["cubic spline"])):
        a1.semilogy(df["nodes"], df[col], "-o", color=c, label=label)
    a1.set_xlabel("equispaced nodes used")
    a1.set_ylabel("RMSE on unused years (°C)")
    a1.set_title("Global polynomial diverges")
    a1.legend()
    a2.semilogy(df["nodes"], df["cond_vandermonde"], "-o", color=METHOD_COLOR["lagrange (global)"])
    a2.axhline(1 / 2.2e-16, color=INK_2, lw=0.8, ls="--")
    a2.text(df["nodes"].iloc[0], 1 / 2.2e-16 * 2, "1/machine-epsilon: no digits left", fontsize=8, color=INK_2)
    a2.set_xlabel("equispaced nodes used")
    a2.set_ylabel("cond(V)")
    a2.set_title("Vandermonde conditioning")
    return fig


def plot_scree(s, max_components: int = 8):
    ev = explained_variance(s)[:max_components]
    x = np.arange(1, len(ev) + 1)
    fig, ax = new_fig(w=6, h=3.6)
    ax.bar(x, ev, color=BLUE, width=0.6, label="component")
    ax.plot(x, np.cumsum(ev), "-o", color=ORANGE, label="cumulative")
    for xi, v, c in zip(x[:4], ev[:4], np.cumsum(ev)[:4]):
        ax.annotate(f"{c:.1%}", (xi, c), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("component")
    ax.set_ylabel("share of variance")
    ax.set_title("Explained variance")
    ax.legend(loc="center right")
    return fig


def plot_modes(res, names, years, n_modes: int = 3):
    fig, axes = new_fig(ncols=2, nrows=n_modes, w=5.4, h=2.3, gridspec_kw={"width_ratios": [1, 1.7]})
    ev = explained_variance(res.s)
    for i in range(n_modes):
        pat = spatial_pattern(res, i)
        axes[i, 0].bar(range(len(names)), pat, color=[BLUE if v >= 0 else ORANGE for v in pat], width=0.65)
        axes[i, 0].axhline(0, color=INK_2, lw=0.8)
        axes[i, 0].set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7.5)
        axes[i, 0].set_ylim(-1.15, 1.15)
        axes[i, 0].set_title(f"Mode {i + 1} pattern ({ev[i]:.1%})")
        axes[i, 1].plot(years, temporal_amplitude(res, i), color=BLUE, lw=1.6)
        axes[i, 1].axhline(0, color=INK_2, lw=0.8)
        axes[i, 1].set_title(f"Mode {i + 1} time series")
    axes[-1, 1].set_xlabel("year")
    return fig


def plot_original_vs_reconstructed(A, Ak, years, names, k: int):
    """Original | rank-k reconstruction | residual, shared diverging scale."""
    fig, axes = new_fig(ncols=3, w=4.4, h=3.4, sharey=True)
    lim = np.max(np.abs(A - A.mean(axis=1, keepdims=True)))
    ext = [years[0], years[-1], len(names) - 0.5, -0.5]
    for ax, M, title in zip(axes, (A, Ak, A - Ak), ("Original", f"Rank-{k} reconstruction", "Residual")):
        base = M if title != "Residual" else M
        im = ax.imshow(base, cmap=DIV, vmin=-lim * 1.6, vmax=lim * 1.6, aspect="auto", extent=ext, interpolation="nearest")
        ax.set_title(title)
        ax.grid(False)
        ax.set_xlabel("year")
    axes[0].set_yticks(range(len(names)), names)
    fig.colorbar(im, ax=axes, shrink=0.8, label="anomaly (°C)")
    return fig


def plot_error_vs_rank(rank_df, held_out=None):
    """In-sample rank error (and optionally held-out completion error)."""
    fig, ax = new_fig(w=6, h=3.6)
    ax.plot(rank_df.index, rank_df["rmse"], "-o", color=BLUE, label="in-sample RMSE (all entries)")
    if held_out is not None:
        ax.plot(held_out.index, held_out["held_out_rmse"], "-o", color=ORANGE, label="held-out RMSE (hidden entries)")
    ax.set_xlabel("retained rank k")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Reconstruction error vs rank")
    ax.legend()
    return fig

# ----------------------------------------------------------------------------
# Run this file
# ----------------------------------------------------------------------------


def main():
    """Person 1's deliverables: the reconstruction graph, the Runge test, and the SVD results."""
    df, years, values = load_global()

    # Part A: hide 20% of the years, rebuild them, compare with the truth
    hidden = make_hidden_mask(len(years), round(0.2 * len(years)), 1, np.random.default_rng(7))
    preds = predict_hidden(years, values, hidden)
    save_figure(plot_reconstruction(years, values, hidden, preds), "interpolation_reconstruction")
    runge = runge_experiment(years, values)
    save_table(runge, "interpolation_runge", index=False)
    save_figure(plot_runge(runge), "interpolation_runge")

    # Part B: SVD of the region-by-year matrix
    A, ayears, names = region_matrix(df)
    res = compute_svd(A, "row")
    ev = explained_variance(res.s)
    variance = pd.DataFrame({"explained": ev, "cumulative": np.cumsum(ev), "singular_value": res.s},
                            index=pd.RangeIndex(1, len(res.s) + 1, name="component"))
    save_table(variance, "svd_variance")
    table = rank_table(A, res)
    save_table(table, "svd_rank_table")
    save_figure(plot_scree(res.s), "svd_scree")
    save_figure(plot_modes(res, names, ayears), "svd_modes")
    save_figure(plot_original_vs_reconstructed(A, reconstruct(res, 2), ayears, names, 2), "svd_reconstruction_rank2")
    print(table.round(3))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# Tests  (python -m pytest person1_math.py)
# ----------------------------------------------------------------------------


import pytest  # noqa: E402  (only the tests need these two)
from scipy.interpolate import CubicSpline  # noqa: E402


def test_linear_matches_numpy():
    x = np.array([0, 1, 3, 4, 8.0]); y = np.array([1, 0, 2, 5, 3.0]); q = np.linspace(0, 8, 33)
    assert np.allclose(linear_interpolate(x, y, q), np.interp(q, x, y))


def test_global_lagrange_reproduces_polynomial_exactly():
    x = np.linspace(0, 10, 6)
    f = lambda t: 0.5 * t**3 - 2 * t**2 + t + 4
    q = np.linspace(0, 10, 21)
    assert np.allclose(lagrange_interpolate(x, f(x), q), f(q))


def test_local_lagrange_window2_is_linear_and_window4_exact_for_cubics():
    x = np.arange(0, 30.0); y = x**3 / 100
    q = np.array([3.5, 10.2, 20.7])
    assert np.allclose(lagrange_interpolate(x, y, q, window=2), linear_interpolate(x, y, q))
    assert np.allclose(lagrange_interpolate(x, y, q, window=4), q**3 / 100)


def test_lagrange_hits_nodes():
    x = np.array([1.0, 2, 4, 7]); y = np.array([3.0, -1, 2, 5])
    assert np.allclose(lagrange_interpolate(x, y, x), y)


def test_vandermonde_agrees_with_lagrange_for_few_nodes():
    x = np.linspace(1900, 2000, 6); y = np.array([0.1, -0.2, 0.0, 0.3, 0.5, 0.4]); q = np.linspace(1900, 2000, 11)
    assert np.allclose(vandermonde_interpolate(x, y, q), lagrange_interpolate(x, y, q))


def test_vandermonde_condition_grows_fast():
    c = [vandermonde_condition(np.arange(n, dtype=float)) for n in (5, 10, 20)]
    assert c[0] < c[1] < c[2] and c[2] > 1e7


def test_natural_spline_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.sort(rng.choice(np.arange(1880, 2020), 25, replace=False)).astype(float); y = rng.normal(size=25)
    q = np.linspace(x[0], x[-1], 200)
    assert np.allclose(cubic_spline_interpolate(x, y, q), CubicSpline(x, y, bc_type="natural")(q))


def test_spline_has_zero_end_curvature():
    x = np.arange(8.0); y = np.sin(x)
    M = natural_spline_second_derivatives(x, y)
    assert M[0] == 0 and M[-1] == 0


def test_no_extrapolation_and_input_validation():
    with pytest.raises(ValueError):
        linear_interpolate([0, 1, 2], [0, 1, 2], [3])
    with pytest.raises(ValueError):
        linear_interpolate([0, 2, 1], [0, 1, 2], [1])


def test_hidden_mask_properties():
    rng = np.random.default_rng(0)
    for g in (1, 3, 5):
        m = make_hidden_mask(100, 20, g, rng)
        assert not m[0] and not m[-1] and m.sum() > 0
        # runs have length g, separated by known points
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])))[::2]
        assert set(runs) == {g}


def test_runge_experiment_skips_node_counts_larger_than_series():
    x = np.arange(1990.0, 2026); y = np.sin(x / 5)  # 36 points, default counts go up to 40
    df = runge_experiment(x, y)
    assert len(df) > 0 and df["nodes"].max() < len(x)


def _A(seed=0, shape=(8, 60)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(shape[0], 3)) @ rng.normal(size=(3, shape[1])) + rng.normal(size=(shape[0], 1))


def test_full_rank_reconstruction_is_exact():
    A = _A(); r = compute_svd(A)
    assert np.allclose(reconstruct(r, r.rank), A)


def test_eckart_young_error_equals_tail_singular_values():
    A = np.random.default_rng(1).normal(size=(8, 50)); r = compute_svd(A)
    t = rank_table(A, r)
    assert np.allclose(t["frob_error"], t["eckart_young"])


def test_error_decreases_monotonically_with_rank():
    A = np.random.default_rng(2).normal(size=(8, 50)); t = rank_table(A, compute_svd(A))
    assert (np.diff(t["frob_error"]) <= 1e-12).all()


def test_explained_variance_sums_to_one_and_sorted():
    r = compute_svd(_A()); ev = explained_variance(r.s)
    assert np.isclose(ev.sum(), 1) and (np.diff(ev) <= 0).all()


def test_exact_low_rank_has_numerical_rank():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 40))
    assert numerical_rank(np.linalg.svd(A, compute_uv=False)) == 3


def test_centering_modes():
    A = _A()
    assert np.allclose(center(A, "row")[0].mean(axis=1), 0)
    assert np.allclose(center(A, "col")[0].mean(axis=0), 0)


def test_weighted_svd_round_trips():
    A = _A(); w = np.sqrt(np.linspace(0.5, 2, 8)); r = compute_svd(A, "row", w)
    assert np.allclose(reconstruct(r, r.rank), A)


def test_hard_impute_recovers_exact_low_rank_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 146)) + rng.normal(size=(8, 1))
    m = random_mask(A.shape, 0.2, rng)
    X = impute_lowrank(np.where(m, np.nan, A), 2)
    assert np.sqrt(np.mean((X[m] - A[m]) ** 2)) < 1e-4


def test_impute_keeps_observed_entries_and_k0_is_row_mean():
    rng = np.random.default_rng(1); A = _A(shape=(8, 40)); m = random_mask(A.shape, 0.2, rng)
    Ao = np.where(m, np.nan, A)
    X = impute_lowrank(Ao, 0)
    assert np.allclose(X[~m], A[~m])
    assert np.allclose(X[m], np.broadcast_to(np.nanmean(Ao, axis=1, keepdims=True), A.shape)[m])


def test_soft_impute_with_large_lambda_collapses_to_row_mean():
    rng = np.random.default_rng(2); A = _A(shape=(8, 40)); m = random_mask(A.shape, 0.2, rng)
    Ao = np.where(m, np.nan, A)
    assert np.allclose(impute_soft(Ao, 1e6), impute_lowrank(Ao, 0))


def test_masks_spare_edges():
    rng = np.random.default_rng(0)
    assert not random_mask((8, 50), 0.5, rng)[:, [0, -1]].any()
    b = block_mask((8, 50), 5, 2, rng)
    assert not b[:, [0, -1]].any() and (b.sum(axis=1) == 10).all()

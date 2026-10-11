"""Part 4: head-to-head evaluation of all methods and numerical stability checks.

Everything here scores predictions only on data the method never saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpolation as ip
from . import metrics
from . import regression as rg
from . import svd_analysis as sv


# ------------------------------------------------------------- interpolation
def interpolation_benchmark(years, values, gap_lengths=(1, 3, 5, 10), frac: float = 0.2,
                            n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean RMSE / MAE of each interpolation method for each gap length."""
    out = []
    for g in gap_lengths:
        res = ip.holdout_experiment(years, values, frac=frac, gap_length=g, n_trials=n_trials, seed=seed)
        s = res.groupby("method")[["rmse", "mae"]].mean().reset_index()
        s["gap_length"] = g
        out.append(s)
    return pd.concat(out, ignore_index=True)


def noise_amplification(x, y, sigma: float = 0.05, n_trials: int = 50, seed: int = 0,
                        methods=None, global_nodes: int = 25) -> pd.DataFrame:
    """Stability of interpolation: perturb the data and see how the curve moves.

    Adds N(0, sigma^2) noise to the known values and reports
    max|interpolant change| / max|noise| on a fine grid. A value near 1 means
    noise passes through unchanged; a large value means the method amplifies it.
    The global Lagrange polynomial uses ``global_nodes`` equispaced nodes.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    grid = np.linspace(x[0], x[-1], 400)
    methods = dict(ip.METHODS) if methods is None else dict(methods)
    methods["lagrange (global)"] = ip.lagrange_interpolate
    rows = []
    for name, f in methods.items():
        if name == "lagrange (global)":
            idx = np.unique(np.round(np.linspace(0, len(x) - 1, global_nodes)).astype(int))
            xs, ys = x[idx], y[idx]
        else:
            xs, ys = x, y
        base = f(xs, ys, grid)
        ratios = []
        for _ in range(n_trials):
            noise = rng.normal(0, sigma, len(xs))
            ratios.append(np.max(np.abs(f(xs, ys + noise, grid) - base)) / np.max(np.abs(noise)))
        rows.append({"method": name, "nodes": len(xs), "amplification": float(np.mean(ratios))})
    return pd.DataFrame(rows).set_index("method")


# ------------------------------------------------------------- extrapolation
def extrapolation_benchmark(t, y, degrees=(1, 2, 3), horizon: int = 10, min_train: int = 40,
                            window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin backtest; returns (raw backtest, summary by degree and lead)."""
    bt = rg.rolling_origin(t, y, degrees, horizon, min_train, window)
    return bt, rg.summarize_backtest(bt, ["degree", "lead"]).reset_index()


def training_period_sensitivity(t, y, degrees=(1, 2), starts=(1880, 1900, 1920, 1940, 1960, 1980),
                                cutoffs=(1990, 2000, 2005, 2010, 2015), horizon: int = 10) -> pd.DataFrame:
    """RMSE of the next ``horizon`` years as the training window start and end move."""
    t, y = np.asarray(t), np.asarray(y)
    rows = []
    for d in degrees:
        for c in cutoffs:
            test = (t > c) & (t <= c + horizon)
            if test.sum() < horizon:
                continue
            for s0 in starts:
                tr = (t >= s0) & (t <= c)
                if tr.sum() <= d + 3:
                    continue
                p = rg.fit_polynomial(t[tr], y[tr], d).predict(t[test])
                rows.append({"degree": d, "start": s0, "cutoff": c, "rmse": metrics.rmse(y[test], p), "n_train": int(tr.sum())})
    return pd.DataFrame(rows)


def forecast_spread(t, y, target_year: int = 2040, degrees=(1, 2), starts=(1880, 1900, 1920, 1940, 1960, 1980)) -> pd.DataFrame:
    """Extrapolated value at ``target_year`` from different training-period starts."""
    t, y = np.asarray(t), np.asarray(y)
    rows = []
    for d in degrees:
        for s0 in starts:
            m = t >= s0
            if m.sum() <= d + 3:  # window starts too close to the end of the data
                continue
            f = rg.fit_polynomial(t[m], y[m], d)
            rows.append({"degree": d, "start": s0, "n_train": int(m.sum()),
                         "forecast": float(f.predict([target_year])[0]),
                         "slope_per_decade_at_end": float(f.slope([t[-1]])[0] * 10)})
    return pd.DataFrame(rows)


# --------------------------------------------------- matrix reconstruction
def _time_interp_rows(A, mask, years, fn):
    """Reconstruct hidden entries of each region using only that region's own time series."""
    out = A.copy()
    for r in range(A.shape[0]):
        known = ~mask[r]
        if mask[r].any():
            out[r, mask[r]] = fn(years[known], A[r, known], years[mask[r]])
    return out


def matrix_reconstruction_benchmark(A, years, ranks=(0, 1, 2, 3, 4), soft_lambdas=(1.0,), mask_kind: str = "random",
                                    frac: float = 0.2, gap_length: int = 5, n_gaps: int = 3,
                                    n_trials: int = 30, seed: int = 0) -> pd.DataFrame:
    """Hide entries of the region x year matrix and reconstruct them by
    (a) each region's own time series (linear, cubic spline) and
    (b) low-rank completion using all regions at once (rank 0 = row mean;
        'soft' = singular values shrunk by lambda instead of truncated).

    mask_kind: 'random' hides entries independently; 'block' hides runs of
    ``gap_length`` consecutive years, ``n_gaps`` per region.
    """
    years = np.asarray(years, float)
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        mask = sv.random_mask(A.shape, frac, rng) if mask_kind == "random" else sv.block_mask(A.shape, gap_length, n_gaps, rng)
        truth = A[mask]
        preds = {"linear (time only)": _time_interp_rows(A, mask, years, ip.linear_interpolate)[mask],
                 "cubic spline (time only)": _time_interp_rows(A, mask, years, ip.cubic_spline_interpolate)[mask]}
        A_obs = np.where(mask, np.nan, A)
        for k in ranks:
            label = "row mean (rank 0)" if k == 0 else f"SVD rank {k}"
            preds[label] = sv.impute_lowrank(A_obs, k)[mask]
        for lam in soft_lambdas:
            preds[f"SVD soft (λ={lam:g})"] = sv.impute_soft(A_obs, lam)[mask]
        for name, p in preds.items():
            rows.append({"trial": trial, "method": name, "rmse": metrics.rmse(truth, p), "mae": metrics.mae(truth, p),
                         "hidden_fraction": float(mask.mean())})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, by="method") -> pd.DataFrame:
    g = results.groupby(by)[["rmse", "mae"]]
    return g.mean().join(g.std(), rsuffix="_std")


def rank_generalisation(A, years, max_rank: int = 6, frac: float = 0.2, n_trials: int = 30, seed: int = 0) -> pd.DataFrame:
    """In-sample SVD error vs. held-out completion error as the rank grows.

    In-sample error always falls; held-out error is what tells you when extra
    components start fitting noise.
    """
    full = sv.rank_table(A, sv.compute_svd(A), range(0, min(max_rank, A.shape[0]) + 1))[["rmse"]].rename(columns={"rmse": "in_sample_rmse"})
    bench = matrix_reconstruction_benchmark(A, years, ranks=range(0, min(max_rank, A.shape[0]) + 1), frac=frac,
                                            n_trials=n_trials, seed=seed)
    svd_rows = bench[bench["method"].str.startswith(("SVD rank", "row mean"))].copy()
    svd_rows["k"] = svd_rows["method"].map(lambda m: 0 if m.startswith("row") else int(m.split()[-1]))
    held = svd_rows.groupby("k")["rmse"].mean().rename("held_out_rmse")
    return full.join(held)

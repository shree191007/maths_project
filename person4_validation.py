"""Person 4: VALIDATION + DASHBOARD  (test everything, then put it all in one app)

What this file does
    ERROR SCORES      rmse, mae (plus max_error, bias): how wrong is a guess? Smaller is better.
    THE GOLDEN RULE   only score a method on data it has never seen.
    TESTS
        * interpolation   hide years, rebuild them, score each method      (holdout_experiment)
        * regression      forecast the later years from the earlier ones   (cutoff_holdout, rolling_origin)
        * SVD             hide matrix entries, rebuild them at each rank   (matrix_reconstruction_benchmark)
        * sensitivity     change the training years and see how much the forecast moves
        * stability       add noise to the data and see which method amplifies it
    DASHBOARD         five Streamlit tabs that re-run the real code from all four files

Run it
    python person4_validation.py all       # runs ALL four files and writes every figure and table
    python person4_validation.py           # only this file's tests -> reports/
    streamlit run person4_validation.py    # the interactive dashboard
    python -m pytest person4_validation.py # tests for this file

Needs: person1_math.py, person2_data.py, person3_regression.py
"""
from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("streamlit").setLevel(logging.ERROR)  # quiet "no runtime" notices outside the dashboard
import streamlit as st  # noqa: E402

import person1_math  # noqa: E402
import person2_data  # noqa: E402
import person3_regression  # noqa: E402
from person1_math import (METHOD_COLOR, METHODS, block_mask, compute_svd, cubic_spline_interpolate,  # noqa: E402
                          explained_variance, impute_lowrank, impute_soft, lagrange_interpolate,
                          linear_interpolate, make_hidden_mask, plot_error_vs_rank, plot_modes,
                          plot_original_vs_reconstructed, plot_reconstruction, plot_runge, plot_scree,
                          predict_hidden, random_mask, rank_table, reconstruct, runge_experiment, CENTER_MODES)
from person2_data import (BLUE, CAT, GRAY, INK_2, ORANGE, SEQ, band_area_weights, global_series,  # noqa: E402
                          load_global, load_zonal, new_fig, region_matrix, save_figure, save_table)
from person3_regression import (DEG_COLOR, DEG_NAME, compare_degrees, conditioning_table, design_matrix,  # noqa: E402
                                extrapolate, fit_polynomial, forecast_spread, plot_fits, plot_forecast_spread,
                                split_by_year)


# ----------------------------------------------------------------------------
# 1. Error scores
# ----------------------------------------------------------------------------


def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error, ||e||_2 / sqrt(n)."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, ||e||_1 / n."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def max_error(y_true, y_pred) -> float:
    """Worst-case error, ||e||_inf."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def bias(y_true, y_pred) -> float:
    """Mean signed error (prediction minus truth)."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def summary(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }

# ----------------------------------------------------------------------------
# 2. Test: interpolation on hidden years
# ----------------------------------------------------------------------------


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
                         "rmse": rmse(values[hidden], pred), "mae": mae(values[hidden], pred)})
    return pd.DataFrame(rows)


def summarize_holdout(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of per-trial RMSE / MAE for each method (sorted best first)."""
    g = results.groupby("method")[["rmse", "mae"]]
    out = g.mean().join(g.std(), rsuffix="_std")
    return out.sort_values("rmse")


def interpolation_benchmark(years, values, gap_lengths=(1, 3, 5, 10), frac: float = 0.2,
                            n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean RMSE / MAE of each interpolation method for each gap length."""
    out = []
    for g in gap_lengths:
        res = holdout_experiment(years, values, frac=frac, gap_length=g, n_trials=n_trials, seed=seed)
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
    methods = dict(METHODS) if methods is None else dict(methods)
    methods["lagrange (global)"] = lagrange_interpolate
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

# ----------------------------------------------------------------------------
# 3. Test: regression forecasting later years
# ----------------------------------------------------------------------------


def cutoff_holdout(t, y, cutoff: int, degrees=(1, 2, 3), level: float = 0.95) -> pd.DataFrame:
    """Train up to ``cutoff``, score on every later year; one row per degree."""
    tr_t, tr_y, te_t, te_y = split_by_year(t, y, cutoff)
    rows = []
    for d in degrees:
        f = fit_polynomial(tr_t, tr_y, d)
        pred, lo, hi = f.predict(te_t, level)
        rows.append({"degree": d, "train_rmse": rmse(tr_y, f.predict(tr_t)),
                     "test_rmse": rmse(te_y, pred), "test_mae": mae(te_y, pred),
                     "test_bias": bias(te_y, pred)})
    return pd.DataFrame(rows).set_index("degree")


def rolling_origin(t, y, degrees=(1, 2, 3), horizon: int = 10, min_train: int = 40,
                   window: int | None = None, level: float = 0.95) -> pd.DataFrame:
    """Forward-chaining backtest.

    For every origin year T (from the min_train-th year onward) fit on years <= T
    (only the last ``window`` years if given) and forecast T+1 .. T+horizon.
    Returns one row per (degree, origin, lead).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    rows = []
    for end in range(min_train, len(t) - horizon + 1):
        lo_i = 0 if window is None else max(0, end - window)
        if end - lo_i <= max(degrees) + 1:
            continue
        for d in degrees:
            f = fit_polynomial(t[lo_i:end], y[lo_i:end], d)
            ft, fy = t[end:end + horizon], y[end:end + horizon]
            pred, lo, hi = f.predict(ft, level)
            for k in range(len(ft)):
                rows.append({"degree": d, "origin": int(t[end - 1]), "lead": k + 1, "year": int(ft[k]),
                             "y_true": fy[k], "y_pred": pred[k], "lo": lo[k], "hi": hi[k]})
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame, by: str | list = "degree") -> pd.DataFrame:
    """RMSE, MAE, bias and prediction-interval coverage of a backtest table."""
    def agg(g):
        e = g["y_pred"] - g["y_true"]
        return pd.Series({"rmse": np.sqrt(np.mean(e**2)), "mae": np.mean(np.abs(e)), "bias": np.mean(e),
                          "coverage": np.mean((g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])), "n": len(g)})
    return bt.groupby(by)[["y_true", "y_pred", "lo", "hi"]].apply(agg)


def random_holdout(t, y, degrees=(1, 2, 3), frac: float = 0.2, n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """Hide random interior years, fit on the rest, score on the hidden ones."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        hidden = np.zeros(len(t), bool)
        hidden[rng.choice(np.arange(1, len(t) - 1), round(frac * len(t)), replace=False)] = True
        for d in degrees:
            f = fit_polynomial(t[~hidden], y[~hidden], d)
            p = f.predict(t[hidden])
            rows.append({"trial": trial, "degree": d, "rmse": rmse(y[hidden], p), "mae": mae(y[hidden], p)})
    return pd.DataFrame(rows)


def extrapolation_benchmark(t, y, degrees=(1, 2, 3), horizon: int = 10, min_train: int = 40,
                            window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin backtest; returns (raw backtest, summary by degree and lead)."""
    bt = rolling_origin(t, y, degrees, horizon, min_train, window)
    return bt, summarize_backtest(bt, ["degree", "lead"]).reset_index()


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
                p = fit_polynomial(t[tr], y[tr], d).predict(t[test])
                rows.append({"degree": d, "start": s0, "cutoff": c, "rmse": rmse(y[test], p), "n_train": int(tr.sum())})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# 4. Test: SVD / matrix reconstruction on hidden entries
# ----------------------------------------------------------------------------


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
        mask = random_mask(A.shape, frac, rng) if mask_kind == "random" else block_mask(A.shape, gap_length, n_gaps, rng)
        truth = A[mask]
        preds = {"linear (time only)": _time_interp_rows(A, mask, years, linear_interpolate)[mask],
                 "cubic spline (time only)": _time_interp_rows(A, mask, years, cubic_spline_interpolate)[mask]}
        A_obs = np.where(mask, np.nan, A)
        for k in ranks:
            label = "row mean (rank 0)" if k == 0 else f"SVD rank {k}"
            preds[label] = impute_lowrank(A_obs, k)[mask]
        for lam in soft_lambdas:
            preds[f"SVD soft (λ={lam:g})"] = impute_soft(A_obs, lam)[mask]
        for name, p in preds.items():
            rows.append({"trial": trial, "method": name, "rmse": rmse(truth, p), "mae": mae(truth, p),
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
    full = rank_table(A, compute_svd(A), range(0, min(max_rank, A.shape[0]) + 1))[["rmse"]].rename(columns={"rmse": "in_sample_rmse"})
    bench = matrix_reconstruction_benchmark(A, years, ranks=range(0, min(max_rank, A.shape[0]) + 1), frac=frac,
                                            n_trials=n_trials, seed=seed)
    svd_rows = bench[bench["method"].str.startswith(("SVD rank", "row mean"))].copy()
    svd_rows["k"] = svd_rows["method"].map(lambda m: 0 if m.startswith("row") else int(m.split()[-1]))
    held = svd_rows.groupby("k")["rmse"].mean().rename("held_out_rmse")
    return full.join(held)

# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------


def plot_error_comparison(bench):
    """bench: columns method, gap_length, rmse, mae (validation.interpolation_benchmark)."""
    fig, axes = new_fig(ncols=2, w=5.6, h=3.6, sharey=True)
    methods = list(dict.fromkeys(bench["method"]))
    gaps = sorted(bench["gap_length"].unique())
    width = 0.8 / len(methods)
    for ax, metric in zip(axes, ("rmse", "mae")):
        for i, m in enumerate(methods):
            sub = bench[bench["method"] == m].set_index("gap_length").loc[gaps, metric]
            ax.bar(np.arange(len(gaps)) + (i - (len(methods) - 1) / 2) * width, sub.values, width * 0.92,
                   color=METHOD_COLOR.get(m, CAT[i]), label=m)
        ax.set_xticks(range(len(gaps)), [str(g) for g in gaps])
        ax.set_xlabel("consecutive years hidden")
        ax.set_title(metric.upper() + " (°C)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, loc="upper left")
    return fig


def plot_backtest_by_lead(summary):
    """summary: columns degree, lead, rmse, coverage (summarize_backtest(bt, ['degree','lead']))."""
    fig, (a1, a2) = new_fig(ncols=2, w=5.6, h=3.6)
    for d, g in summary.groupby("degree"):
        a1.plot(g["lead"], g["rmse"], "-o", color=DEG_COLOR.get(d, CAT[3]), ms=4, label=DEG_NAME.get(d, d))
        a2.plot(g["lead"], g["coverage"], "-o", color=DEG_COLOR.get(d, CAT[3]), ms=4, label=DEG_NAME.get(d, d))
    a1.set_title("Forecast RMSE by lead time")
    a1.set_xlabel("years ahead")
    a1.set_ylabel("RMSE (°C)")
    a1.legend()
    a2.axhline(0.95, color=INK_2, lw=0.8, ls="--")
    a2.text(1, 0.955, "nominal 95%", fontsize=8, color=INK_2)
    a2.set_ylim(0, 1.02)
    a2.set_title("Prediction-interval coverage")
    a2.set_xlabel("years ahead")
    return fig


def plot_training_sensitivity(df):
    """Heatmaps (train start x forecast origin) of 10-year-ahead RMSE, one panel per degree."""
    degs = sorted(df["degree"].unique())
    fig, axes = new_fig(ncols=len(degs), w=4.6, h=3.6, squeeze=False)
    vmax = df["rmse"].max()
    for ax, d in zip(axes[0], degs):
        p = df[df["degree"] == d].pivot(index="start", columns="cutoff", values="rmse").sort_index()
        im = ax.imshow(p.values, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(p.columns)), p.columns)
        ax.set_yticks(range(len(p.index)), p.index)
        ax.grid(False)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > vmax * .55 else "#0b0b0b")
        ax.set_title(f"{DEG_NAME.get(d, d)}: RMSE (°C)")
        ax.set_xlabel("training ends")
    axes[0][0].set_ylabel("training starts")
    return fig


def _color(method: str) -> str:
    if method.startswith("SVD"):
        return ORANGE
    if method.startswith("row mean"):
        return GRAY
    return BLUE


def plot_method_comparison(summary, title="Held-out reconstruction error"):
    """summary: index = method, columns rmse, mae, rmse_std (validation.summarize)."""
    s = summary.sort_values("rmse", ascending=False)
    fig, ax = new_fig(w=7, h=0.42 * len(s) + 1.4)
    y = np.arange(len(s))
    ax.barh(y, s["rmse"], xerr=s["rmse_std"], color=[_color(m) for m in s.index], height=0.6,
            error_kw={"ecolor": INK_2, "lw": 1, "capsize": 2})
    for yi, v, e in zip(y, s["rmse"], s["rmse_std"]):
        ax.text(v + e + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, (s["rmse"] + s["rmse_std"]).max() * 1.12)
    ax.set_xlabel("RMSE (°C), mean ± std over trials")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return fig


def plot_noise_amplification(df):
    fig, ax = new_fig(w=6, h=3.4)
    d = df.sort_values("amplification")
    ax.barh(range(len(d)), d["amplification"], color=[CAT[1] if "global" in m else BLUE for m in d.index], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(range(len(d)), d.index)
    ax.axvline(1, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel("max |change in curve| / max |noise| (log scale; 1 = no amplification)")
    ax.set_title("Sensitivity to noise in the data")
    ax.grid(axis="y", visible=False)
    return fig

# ----------------------------------------------------------------------------
# 6. Dashboard (streamlit run person4_validation.py)
# ----------------------------------------------------------------------------


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data
def cached_zonal():
    return load_zonal()


@st.cache_data
def cached_interp_benchmark(start, end, n_trials):
    g = global_series(cached_zonal(), start, end)
    return interpolation_benchmark(g.index.to_numpy(float), g.to_numpy(), n_trials=n_trials)


@st.cache_data
def cached_runge(start, end):
    g = global_series(cached_zonal(), start, end)
    return runge_experiment(g.index.to_numpy(float), g.to_numpy())


@st.cache_data
def cached_backtest(start, end, horizon, min_train, window):
    g = global_series(cached_zonal(), start, end)
    bt, summ = extrapolation_benchmark(g.index.to_numpy(float), g.to_numpy(), horizon=horizon, min_train=min_train, window=window)
    return summarize_backtest(bt, "degree"), summ


@st.cache_data
def cached_matrix_benchmark(start, end, kind, frac, gap, n_gaps, n_trials):
    A, yrs, _ = region_matrix(cached_zonal(), start, end)
    b = matrix_reconstruction_benchmark(A, yrs, mask_kind=kind, frac=frac, gap_length=gap, n_gaps=n_gaps, n_trials=n_trials)
    return summarize(b)


@st.cache_data
def cached_generalisation(start, end, n_trials):
    A, yrs, _ = region_matrix(cached_zonal(), start, end)
    return rank_generalisation(A, yrs, n_trials=n_trials)


@st.cache_data
def cached_sensitivity(start, end, horizon):
    g = global_series(cached_zonal(), start, end)
    t, y = g.index.to_numpy(float), g.to_numpy()
    return training_period_sensitivity(t, y, horizon=horizon), forecast_spread(t, y)


@st.cache_data
def cached_noise(start, end, sigma):
    g = global_series(cached_zonal(), start, end)
    return noise_amplification(g.index.to_numpy(float), g.to_numpy(), sigma=sigma)


def dashboard():
    """The Streamlit app: run with  streamlit run person4_validation.py"""
    st.set_page_config(page_title="ClimateMatrix", page_icon="🌡️", layout="wide")

    # ------------------------------------------------------------------ sidebar
    df = cached_zonal()
    st.sidebar.title("ClimateMatrix")
    st.sidebar.caption("NASA GISTEMP v4 zonal annual anomalies (°C vs 1951–1980)")
    y_min, y_max = int(df.index.min()), int(df.index.max())
    start, end = st.sidebar.slider("Study period", y_min, y_max, (y_min, y_max))
    if end - start < 40:
        st.error("Choose a study period of at least 40 years: the hold-out and backtest analyses need enough history.")
        st.stop()
    glob = global_series(df, start, end)
    years, values = glob.index.to_numpy(float), glob.to_numpy()
    A, ayears, names = region_matrix(df, start, end)
    st.sidebar.markdown(f"**{len(years)}** annual values · matrix **{A.shape[0]} × {A.shape[1]}**")
    st.sidebar.markdown("---")
    st.sidebar.caption("Part 1 interpolation · Part 2 regression · Part 3 SVD · Part 4 validation")

    st.title("ClimateMatrix")
    st.markdown("*How effectively can interpolation and low-rank matrix approximation reconstruct missing climate data, "
                "and how reliable is extrapolation beyond the observed period?*")

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Overview", "1 · Interpolation", "2 · Regression", "3 · SVD", "4 · Validation"])

    # ----------------------------------------------------------------- overview
    with tab0:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = new_fig(h=3.6)
            ax.plot(years, values, color=BLUE)
            ax.set_title("Global annual temperature anomaly")
            ax.set_ylabel("°C vs 1951–1980")
            show(fig)
        with c2:
            st.subheader("The data")
            st.write("The matrix **A** has one row per latitude band and one column per year; each entry is that band's "
                     "annual anomaly. The same data are used as eight time series (Parts 1–2) and as one matrix (Part 3).")
            st.dataframe(pd.DataFrame(A, index=names, columns=ayears.astype(int)).iloc[:, -8:].style.format("{:.2f}"))
            st.caption("Last 8 columns of A.")
        st.info("GISTEMP's annual zonal table has **no missing values**, so every 'missing' observation in this app is "
                "hidden on purpose and the hidden truth is used to score each method.")

    # ------------------------------------------------------------- part 1
    with tab1:
        st.subheader("Hide observations, reconstruct them, compare")
        c1, c2, c3 = st.columns(3)
        frac = c1.slider("Fraction hidden", 0.05, 0.4, 0.2, 0.05)
        gap = c2.select_slider("Consecutive years per gap", [1, 2, 3, 5, 10], value=1)
        seed = c3.number_input("Random seed", 0, 9999, 7)
        chosen = st.multiselect("Methods", list(METHODS), default=list(METHODS))
        hidden = make_hidden_mask(len(years), round(frac * len(years)), gap, np.random.default_rng(int(seed)))
        if chosen:
            preds = predict_hidden(years, values, hidden, {m: METHODS[m] for m in chosen})
            show(plot_reconstruction(years, values, hidden, preds, curves=tuple(chosen[:2]) if len(chosen) > 1 else tuple(chosen)))
            st.dataframe(pd.DataFrame({m: summary(values[hidden], p) for m, p in preds.items()}).T.style.format("{:.3f}"))
            st.caption(f"{int(hidden.sum())} hidden years. Curves drawn for the first two selected methods.")
        with st.expander("Benchmark over many random masks", expanded=True):
            n_tr = st.slider("Trials", 20, 300, 100, 20, key="p1trials")
            bench = cached_interp_benchmark(start, end, n_tr)
            show(plot_error_comparison(bench))
        with st.expander("Why not one polynomial through every point? (Runge / conditioning)"):
            r = cached_runge(start, end)
            show(plot_runge(r))
            st.dataframe(r.style.format({"cond_vandermonde": "{:.2e}", "rmse_lagrange_global": "{:.3g}", "rmse_linear": "{:.3f}", "rmse_spline": "{:.3f}"}))

    # ------------------------------------------------------------- part 2
    with tab2:
        st.subheader("Least-squares trends and extrapolation")
        c1, c2, c3 = st.columns(3)
        degs = c1.multiselect("Model degree", [1, 2, 3], default=[1, 2], format_func=lambda d: {1: "linear", 2: "quadratic", 3: "cubic"}[d])
        horizon_year = c2.slider("Extrapolate to", 2030, 2060, 2040)
        level = c3.select_slider("Prediction interval", [0.8, 0.9, 0.95, 0.99], value=0.95)
        if degs:
            fits = [fit_polynomial(years, values, d) for d in degs]
            show(plot_fits(years, values, fits, horizon_year, level))
            st.caption("Bands assume independent errors; annual residuals are autocorrelated, so they are too narrow "
                       "(see backtest coverage below).")
            target = st.multiselect("Report years", list(range(2026, 2061)), default=[2030, 2040])
            if target:
                rows = []
                for f in fits:
                    e = extrapolate(f, target, level)
                    e.insert(0, "model", {1: "linear", 2: "quadratic", 3: "cubic"}[f.degree])
                    rows.append(e)
                st.dataframe(pd.concat(rows, ignore_index=True).style.format({"prediction": "{:.2f}", "lo": "{:.2f}", "hi": "{:.2f}"}), hide_index=True)
            with st.expander("Design matrix, coefficients and conditioning"):
                for f in fits:
                    st.code(f.equation())
                X = design_matrix(years[:5], max(degs), fits[0].t_center, fits[0].t_scale)
                st.write("First 5 rows of X for the highest degree chosen: columns are 1, τ, τ², …")
                st.dataframe(pd.DataFrame(X).style.format("{:.4f}"))
                st.dataframe(compare_degrees(years, values).style.format("{:.3f}"))
                st.dataframe(conditioning_table(years, values).style.format({"cond(X)": "{:.1e}", "cond(XtX)": "{:.1e}", "fit_err": "{:.1e}"}), hide_index=True)
                st.caption("fit_err = max difference in fitted values from the centred-QR reference. Normal equations on raw years lose all accuracy.")
        st.subheader("Holdout: train on the past, forecast the rest")
        cutoff = st.slider("Training ends in", int(years[0]) + 30, int(years[-1]) - 5, min(2005, int(years[-1]) - 5))
        st.dataframe(cutoff_holdout(years, values, cutoff).style.format("{:.3f}"))
        h1, h2, h3 = st.columns(3)
        bt_h = h1.slider("Forecast horizon (years)", 3, 20, 10)
        bt_min = h2.slider("Minimum training years", 20, 80, 40)
        bt_win = h3.select_slider("Training window", ["all"] + [30, 50, 70], value="all")
        if len(years) - bt_min >= bt_h + 5:
            summ_deg, summ_lead = cached_backtest(start, end, bt_h, bt_min, None if bt_win == "all" else bt_win)
            show(plot_backtest_by_lead(summ_lead))
            st.dataframe(summ_deg.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "bias": "{:+.3f}", "coverage": "{:.0%}", "n": "{:.0f}"}))
        else:
            st.warning("Study period too short for this backtest; widen it or lower the minimum training years.")

    # ------------------------------------------------------------- part 3
    with tab3:
        st.subheader("SVD of the region × year matrix")
        c1, c2, c3 = st.columns(3)
        cmode = c1.selectbox("Centring", CENTER_MODES, format_func={"row": "subtract each region's mean", "col": "subtract each year's mean", "none": "none"}.get)
        weighted = c2.checkbox("Area-weight rows (√ band area)", value=False)
        w = np.sqrt(band_area_weights()) if weighted else None
        res = compute_svd(A, cmode, w)
        k = c3.slider("Rank k", 0, res.rank, 2)
        ev = explained_variance(res.s)
        m1, m2, m3 = st.columns(3)
        m1.metric("Mode 1 variance", f"{ev[0]:.1%}")
        m2.metric(f"Variance in rank {k}", f"{ev[:k].sum():.1%}")
        m3.metric("Condition number σ₁/σ_r", f"{res.s[0] / res.s[-1]:.0f}")
        cc1, cc2 = st.columns(2)
        with cc1:
            show(plot_scree(res.s))
        with cc2:
            rt = rank_table(A, res)
            show(plot_error_vs_rank(rt))
        show(plot_original_vs_reconstructed(A, reconstruct(res, k) if k else np.broadcast_to(res.mean, A.shape), ayears, names, k))
        show(plot_modes(res, names, ayears, n_modes=3))
        st.dataframe(rt.style.format({"explained_cum": "{:.1%}", "rel_error": "{:.1%}", "frob_error": "{:.3f}", "eckart_young": "{:.3f}", "rmse": "{:.3f}", "mae": "{:.3f}"}))
        st.caption("frob_error equals the Eckart–Young value √Σ_{i>k} σᵢ² exactly (unweighted case): the SVD truncation is the best rank-k fit.")
        st.markdown("---")
        st.subheader("Reconstruct hidden entries from the other regions")
        d1, d2, d3, d4 = st.columns(4)
        kind = d1.selectbox("Gap type", ["random", "block"])
        if kind == "random":
            hid = d2.slider("Hidden fraction", 0.05, 0.4, 0.2, 0.05)
        else:
            hid = d2.slider("Gap length (years)", 2, 15, 5)
        method = d3.selectbox("Completion", ["hard rank-k", "soft threshold"])
        if method == "hard rank-k":
            kk = d4.slider("Rank k", 1, 6, 1)
        else:
            kk = d4.slider("Shrinkage λ", 0.0, 4.0, 1.0, 0.25)
        rng = np.random.default_rng(3)
        mask = random_mask(A.shape, hid, rng) if kind == "random" else block_mask(A.shape, int(hid), 2, rng)
        A_obs = np.where(mask, np.nan, A)
        filled = impute_lowrank(A_obs, int(kk)) if method == "hard rank-k" else impute_soft(A_obs, float(kk))
        e1, e2, e3 = st.columns(3)
        e1.metric("Hidden entries", int(mask.sum()))
        e2.metric("RMSE on hidden", f"{rmse(A[mask], filled[mask]):.3f} °C")
        e3.metric("MAE on hidden", f"{mae(A[mask], filled[mask]):.3f} °C")
        reg = st.selectbox("Region to display", names, index=0)
        r_i = names.index(reg)
        fig, ax = new_fig(h=3.4)
        ax.plot(ayears, A[r_i], color=GRAY, lw=1.2, label="truth")
        ax.plot(ayears[mask[r_i]], filled[r_i, mask[r_i]], "o", color=ORANGE, label="reconstructed")
        ax.plot(ayears[mask[r_i]], A[r_i, mask[r_i]], "o", mfc="none", mec="#0b0b0b", ms=7, label="hidden truth")
        ax.set_title(f"{reg}: reconstructed vs hidden"); ax.set_ylabel("°C"); ax.legend()
        show(fig)

    # ------------------------------------------------------------- part 4
    with tab4:
        st.subheader("Head-to-head on withheld data")
        st.caption("Every number below is measured on entries a method never saw.")
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox("Mask", ["random", "block"], key="p4kind")
        frac4 = c2.slider("Hidden fraction", 0.05, 0.4, 0.2, 0.05, key="p4frac")
        gap4 = c3.slider("Gap length (block)", 2, 10, 5, key="p4gap")
        trials4 = c4.slider("Trials", 5, 60, 20, 5, key="p4tr")
        summ = cached_matrix_benchmark(start, end, kind, frac4, gap4, 3, trials4)
        show(plot_method_comparison(summ, f"Region×year matrix, {kind} gaps: held-out RMSE"))
        st.dataframe(summ.style.format("{:.3f}"))
        st.markdown("**Rank vs. generalisation**")
        show(plot_error_vs_rank(rank_table(A, compute_svd(A), range(0, 7)), cached_generalisation(start, end, 15)))
        st.markdown("---")
        st.subheader("Sensitivity to the training period")
        hz = st.slider("Forecast horizon (years)", 5, 15, 10, key="p4hz")
        sens, spread = cached_sensitivity(start, end, hz)
        if len(sens):
            show(plot_training_sensitivity(sens))
        else:
            st.warning("Study period too short for this analysis.")
        if len(spread):
            show(plot_forecast_spread(spread))
        st.dataframe(spread.style.format({"forecast": "{:.2f}", "slope_per_decade_at_end": "{:.3f}"}), hide_index=True)
        st.markdown("---")
        st.subheader("Numerical stability")
        sig = st.select_slider("Noise σ (°C)", [0.01, 0.02, 0.05, 0.1], value=0.05)
        show(plot_noise_amplification(cached_noise(start, end, sig)))


# ----------------------------------------------------------------------------
# 7. Run this file
# ----------------------------------------------------------------------------


def main(args: list[str] | None = None):
    """Person 4's deliverables. With the argument 'all' it first runs the other three people's files."""
    args = args or []
    if "all" in args:
        person2_data.main()
        person1_math.main()
        person3_regression.main()

    df, years, values = load_global()
    A, ayears, names = region_matrix(df)

    # interpolation: hide years, rebuild them, score each method
    bench = interpolation_benchmark(years, values, n_trials=200)
    rmse_t = bench.pivot(index="method", columns="gap_length", values="rmse")
    rmse_t.columns = [f"RMSE, gap={c}" for c in rmse_t.columns]
    mae_t = bench.pivot(index="method", columns="gap_length", values="mae")
    mae_t.columns = [f"MAE, gap={c}" for c in mae_t.columns]
    save_table(rmse_t.join(mae_t).sort_values(rmse_t.columns[0]), "validation_interpolation_errors")
    save_figure(plot_error_comparison(bench), "validation_interpolation_errors")

    # regression: train on the past, forecast later years
    save_table(cutoff_holdout(years, values, 2005), "validation_regression_holdout_after_2005")
    save_table(random_holdout(years, values).groupby("degree")[["rmse", "mae"]].mean(), "validation_regression_random_holdout")
    backtest, by_lead = extrapolation_benchmark(years, values)
    save_table(summarize_backtest(backtest, "degree"), "validation_regression_backtest")
    save_figure(plot_backtest_by_lead(by_lead), "validation_regression_backtest")

    # SVD: error on the data kept versus error on the data hidden
    generalisation = rank_generalisation(A, ayears)
    save_table(generalisation, "validation_svd_rank_generalisation")
    in_sample = rank_table(A, compute_svd(A, "row"))
    save_figure(plot_error_vs_rank(in_sample.loc[:6], generalisation), "validation_svd_error_vs_rank")

    # whole matrix: per-region interpolation versus low-rank completion
    for kind, kw in (("random", {"frac": 0.2}), ("block", {"gap_length": 5, "n_gaps": 3})):
        table = summarize(matrix_reconstruction_benchmark(A, ayears, mask_kind=kind, n_trials=30, **kw)).sort_values("rmse")
        save_table(table, f"validation_matrix_{kind}")
        save_figure(plot_method_comparison(table, f"Region×year matrix, {kind} gaps: held-out RMSE"), f"validation_matrix_{kind}")
        print(f"\nMatrix completion, {kind} gaps (RMSE, deg C)\n", table[["rmse", "mae"]].round(3))

    # sensitivity to the training period, and to noise
    sens = training_period_sensitivity(years, values)
    save_table(sens, "validation_training_sensitivity", index=False)
    save_figure(plot_training_sensitivity(sens), "validation_training_sensitivity")
    noise = noise_amplification(years, values)
    save_table(noise, "validation_noise_amplification")
    save_figure(plot_noise_amplification(noise), "validation_noise_amplification")
    print("\nDone. Figures: reports/figures   Tables: reports/tables")


def _in_streamlit() -> bool:
    """True when this file is being run by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():
        dashboard()
    else:
        main(sys.argv[1:])


# ----------------------------------------------------------------------------
# Tests  (python -m pytest person4_validation.py)
# ----------------------------------------------------------------------------


def test_metrics():
    y, p = np.array([0.0, 1, 2]), np.array([0.0, 2, 0])
    assert np.isclose(rmse(y, p), np.sqrt(5 / 3))
    assert np.isclose(mae(y, p), 1.0) and max_error(y, p) == 2
    assert np.isclose(bias(y, p), -1 / 3)


def test_noise_amplification_global_lagrange_is_unstable():
    x = np.arange(1880.0, 2026); y = 0.007 * (x - 1880) + 0.1 * np.sin(x / 5)
    amp = noise_amplification(x, y, n_trials=5)["amplification"]
    assert amp["linear"] <= 1.05 and amp["lagrange (global)"] > 100


def test_matrix_benchmark_ranks_models_sensibly_on_low_rank_data():
    rng = np.random.default_rng(0); yrs = np.arange(60.0)
    A = rng.normal(size=(8, 1)) @ np.sin(yrs / 3)[None, :] + 0.02 * rng.normal(size=(8, 60))
    s = summarize(matrix_reconstruction_benchmark(A, yrs, ranks=(0, 1), n_trials=5)).rmse
    assert s["SVD rank 1"] < s["row mean (rank 0)"] / 3


def test_training_sensitivity_and_spread_shapes():
    t = np.arange(1880.0, 2026); y = 0.007 * (t - 1880) ** 1.3 / 5
    assert len(training_period_sensitivity(t, y)) > 10
    assert len(forecast_spread(t, y)) == 12


def test_holdout_experiment_runs_and_linear_exact_on_line():
    x = np.arange(60.0); y = 0.02 * x + 1
    res = holdout_experiment(x, y, n_trials=5, gap_length=3)
    assert res[res.method == "linear"]["rmse"].max() < 1e-12


def _line(n=60, noise=0.0, seed=0):
    t = np.arange(1960.0, 1960 + n)
    rng = np.random.default_rng(seed)
    return t, 0.03 * (t - 1960) - 0.2 + noise * rng.normal(size=n)


def test_rolling_origin_never_uses_future():
    t, y = _line(80, 0.05)
    bt = rolling_origin(t, y, degrees=(1,), horizon=5, min_train=30)
    assert (bt["year"] > bt["origin"]).all() and bt["lead"].between(1, 5).all()

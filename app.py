"""ClimateMatrix dashboard.   Run:  streamlit run app.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from climatematrix import data, interpolation as ip, metrics, regression as rg, svd_analysis as sv, validation as va
from climatematrix import plot_interpolation as p1, plot_regression as p2, plot_svd as p3, plot_validation as p4
from climatematrix.plot_style import BLUE, GRAY, ORANGE, new_fig

st.set_page_config(page_title="ClimateMatrix", page_icon="🌡️", layout="wide")


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data
def load_zonal():
    return data.load_zonal()


@st.cache_data
def cached_interp_benchmark(start, end, n_trials):
    g = data.global_series(load_zonal(), start, end)
    return va.interpolation_benchmark(g.index.to_numpy(float), g.to_numpy(), n_trials=n_trials)


@st.cache_data
def cached_runge(start, end):
    g = data.global_series(load_zonal(), start, end)
    return ip.runge_experiment(g.index.to_numpy(float), g.to_numpy())


@st.cache_data
def cached_backtest(start, end, horizon, min_train, window):
    g = data.global_series(load_zonal(), start, end)
    bt, summ = va.extrapolation_benchmark(g.index.to_numpy(float), g.to_numpy(), horizon=horizon, min_train=min_train, window=window)
    return rg.summarize_backtest(bt, "degree"), summ


@st.cache_data
def cached_matrix_benchmark(start, end, kind, frac, gap, n_gaps, n_trials):
    A, yrs, _ = data.region_matrix(load_zonal(), start, end)
    b = va.matrix_reconstruction_benchmark(A, yrs, mask_kind=kind, frac=frac, gap_length=gap, n_gaps=n_gaps, n_trials=n_trials)
    return va.summarize(b)


@st.cache_data
def cached_generalisation(start, end, n_trials):
    A, yrs, _ = data.region_matrix(load_zonal(), start, end)
    return va.rank_generalisation(A, yrs, n_trials=n_trials)


@st.cache_data
def cached_sensitivity(start, end, horizon):
    g = data.global_series(load_zonal(), start, end)
    t, y = g.index.to_numpy(float), g.to_numpy()
    return va.training_period_sensitivity(t, y, horizon=horizon), va.forecast_spread(t, y)


@st.cache_data
def cached_noise(start, end, sigma):
    g = data.global_series(load_zonal(), start, end)
    return va.noise_amplification(g.index.to_numpy(float), g.to_numpy(), sigma=sigma)


# ------------------------------------------------------------------ sidebar
df = load_zonal()
st.sidebar.title("ClimateMatrix")
st.sidebar.caption("NASA GISTEMP v4 zonal annual anomalies (°C vs 1951–1980)")
y_min, y_max = int(df.index.min()), int(df.index.max())
start, end = st.sidebar.slider("Study period", y_min, y_max, (y_min, y_max))
if end - start < 40:
    st.error("Choose a study period of at least 40 years: the hold-out and backtest analyses need enough history.")
    st.stop()
glob = data.global_series(df, start, end)
years, values = glob.index.to_numpy(float), glob.to_numpy()
A, ayears, names = data.region_matrix(df, start, end)
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
    chosen = st.multiselect("Methods", list(ip.METHODS), default=list(ip.METHODS))
    hidden = ip.make_hidden_mask(len(years), round(frac * len(years)), gap, np.random.default_rng(int(seed)))
    if chosen:
        preds = ip.predict_hidden(years, values, hidden, {m: ip.METHODS[m] for m in chosen})
        show(p1.plot_reconstruction(years, values, hidden, preds, curves=tuple(chosen[:2]) if len(chosen) > 1 else tuple(chosen)))
        st.dataframe(pd.DataFrame({m: metrics.summary(values[hidden], p) for m, p in preds.items()}).T.style.format("{:.3f}"))
        st.caption(f"{int(hidden.sum())} hidden years. Curves drawn for the first two selected methods.")
    with st.expander("Benchmark over many random masks", expanded=True):
        n_tr = st.slider("Trials", 20, 300, 100, 20, key="p1trials")
        bench = cached_interp_benchmark(start, end, n_tr)
        show(p1.plot_error_comparison(bench))
    with st.expander("Why not one polynomial through every point? (Runge / conditioning)"):
        r = cached_runge(start, end)
        show(p1.plot_runge(r))
        st.dataframe(r.style.format({"cond_vandermonde": "{:.2e}", "rmse_lagrange_global": "{:.3g}", "rmse_linear": "{:.3f}", "rmse_spline": "{:.3f}"}))

# ------------------------------------------------------------- part 2
with tab2:
    st.subheader("Least-squares trends and extrapolation")
    c1, c2, c3 = st.columns(3)
    degs = c1.multiselect("Model degree", [1, 2, 3], default=[1, 2], format_func=lambda d: {1: "linear", 2: "quadratic", 3: "cubic"}[d])
    horizon_year = c2.slider("Extrapolate to", 2030, 2060, 2040)
    level = c3.select_slider("Prediction interval", [0.8, 0.9, 0.95, 0.99], value=0.95)
    if degs:
        fits = [rg.fit_polynomial(years, values, d) for d in degs]
        show(p2.plot_fits(years, values, fits, horizon_year, level))
        st.caption("Bands assume independent errors; annual residuals are autocorrelated, so they are too narrow "
                   "(see backtest coverage below).")
        target = st.multiselect("Report years", list(range(2026, 2061)), default=[2030, 2040])
        if target:
            rows = []
            for f in fits:
                e = rg.extrapolate(f, target, level)
                e.insert(0, "model", {1: "linear", 2: "quadratic", 3: "cubic"}[f.degree])
                rows.append(e)
            st.dataframe(pd.concat(rows, ignore_index=True).style.format({"prediction": "{:.2f}", "lo": "{:.2f}", "hi": "{:.2f}"}), hide_index=True)
        with st.expander("Design matrix, coefficients and conditioning"):
            for f in fits:
                st.code(f.equation())
            X = rg.design_matrix(years[:5], max(degs), fits[0].t_center, fits[0].t_scale)
            st.write("First 5 rows of X for the highest degree chosen: columns are 1, τ, τ², …")
            st.dataframe(pd.DataFrame(X).style.format("{:.4f}"))
            st.dataframe(rg.compare_degrees(years, values).style.format("{:.3f}"))
            st.dataframe(rg.conditioning_table(years, values).style.format({"cond(X)": "{:.1e}", "cond(XtX)": "{:.1e}", "fit_err": "{:.1e}"}), hide_index=True)
            st.caption("fit_err = max difference in fitted values from the centred-QR reference. Normal equations on raw years lose all accuracy.")
    st.subheader("Holdout: train on the past, forecast the rest")
    cutoff = st.slider("Training ends in", int(years[0]) + 30, int(years[-1]) - 5, min(2005, int(years[-1]) - 5))
    st.dataframe(rg.cutoff_holdout(years, values, cutoff).style.format("{:.3f}"))
    h1, h2, h3 = st.columns(3)
    bt_h = h1.slider("Forecast horizon (years)", 3, 20, 10)
    bt_min = h2.slider("Minimum training years", 20, 80, 40)
    bt_win = h3.select_slider("Training window", ["all"] + [30, 50, 70], value="all")
    if len(years) - bt_min >= bt_h + 5:
        summ_deg, summ_lead = cached_backtest(start, end, bt_h, bt_min, None if bt_win == "all" else bt_win)
        show(p2.plot_backtest_by_lead(summ_lead))
        st.dataframe(summ_deg.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "bias": "{:+.3f}", "coverage": "{:.0%}", "n": "{:.0f}"}))
    else:
        st.warning("Study period too short for this backtest; widen it or lower the minimum training years.")

# ------------------------------------------------------------- part 3
with tab3:
    st.subheader("SVD of the region × year matrix")
    c1, c2, c3 = st.columns(3)
    cmode = c1.selectbox("Centring", sv.CENTER_MODES, format_func={"row": "subtract each region's mean", "col": "subtract each year's mean", "none": "none"}.get)
    weighted = c2.checkbox("Area-weight rows (√ band area)", value=False)
    w = np.sqrt(data.band_area_weights()) if weighted else None
    res = sv.compute_svd(A, cmode, w)
    k = c3.slider("Rank k", 0, res.rank, 2)
    ev = sv.explained_variance(res.s)
    m1, m2, m3 = st.columns(3)
    m1.metric("Mode 1 variance", f"{ev[0]:.1%}")
    m2.metric(f"Variance in rank {k}", f"{ev[:k].sum():.1%}")
    m3.metric("Condition number σ₁/σ_r", f"{res.s[0] / res.s[-1]:.0f}")
    cc1, cc2 = st.columns(2)
    with cc1:
        show(p3.plot_scree(res.s))
    with cc2:
        rt = sv.rank_table(A, res)
        show(p3.plot_error_vs_rank(rt))
    show(p3.plot_original_vs_reconstructed(A, sv.reconstruct(res, k) if k else np.broadcast_to(res.mean, A.shape), ayears, names, k))
    show(p3.plot_modes(res, names, ayears, n_modes=3))
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
    mask = sv.random_mask(A.shape, hid, rng) if kind == "random" else sv.block_mask(A.shape, int(hid), 2, rng)
    A_obs = np.where(mask, np.nan, A)
    filled = sv.impute_lowrank(A_obs, int(kk)) if method == "hard rank-k" else sv.impute_soft(A_obs, float(kk))
    e1, e2, e3 = st.columns(3)
    e1.metric("Hidden entries", int(mask.sum()))
    e2.metric("RMSE on hidden", f"{metrics.rmse(A[mask], filled[mask]):.3f} °C")
    e3.metric("MAE on hidden", f"{metrics.mae(A[mask], filled[mask]):.3f} °C")
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
    show(p4.plot_method_comparison(summ, f"Region×year matrix, {kind} gaps: held-out RMSE"))
    st.dataframe(summ.style.format("{:.3f}"))
    st.markdown("**Rank vs. generalisation**")
    show(p3.plot_error_vs_rank(sv.rank_table(A, sv.compute_svd(A), range(0, 7)), cached_generalisation(start, end, 15)))
    st.markdown("---")
    st.subheader("Sensitivity to the training period")
    hz = st.slider("Forecast horizon (years)", 5, 15, 10, key="p4hz")
    sens, spread = cached_sensitivity(start, end, hz)
    if len(sens):
        show(p2.plot_training_sensitivity(sens))
    else:
        st.warning("Study period too short for this analysis.")
    if len(spread):
        show(p2.plot_forecast_spread(spread))
    st.dataframe(spread.style.format({"forecast": "{:.2f}", "slope_per_decade_at_end": "{:.3f}"}), hide_index=True)
    st.markdown("---")
    st.subheader("Numerical stability")
    sig = st.select_slider("Noise σ (°C)", [0.01, 0.02, 0.05, 0.1], value=0.05)
    show(p4.plot_noise_amplification(cached_noise(start, end, sig)))

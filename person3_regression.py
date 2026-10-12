"""Person 3: REGRESSION  (trend fitting and looking into the future)

What this file does
    * design_matrix       builds X: one row per year, columns 1, t, t^2, ...
    * fit_polynomial      solves the least-squares problem  min ||X b - y||  (QR, not the inverse)
    * FitResult           the fitted model: coefficients, predict() with 95% interval, slope(), equation()
    * extrapolate         predictions (with intervals) for future years such as 2030 and 2040
    * compare_degrees     straight line vs curve vs cubic on the history
    * conditioning_table  shows why the textbook (X^T X)^-1 X^T y formula fails on raw years
    * forecast_spread     the same model trained on different historical periods, forecasting 2040

Run it
    python person3_regression.py              # writes reports/figures/regression_*.png and tables
    python -m pytest person3_regression.py    # tests for this file

Needs: person2_data.py (data loading and chart style).

Remember: extrapolation (looking beyond the data) is a different task from the
reconstruction in person1_math.py (filling gaps inside the data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr, solve_triangular

from person2_data import CAT, GRAY, INK_2, load_global, new_fig, save_figure, save_table


def _rmse(y_true, y_pred) -> float:
    """Local copy of the RMSE formula, so this file does not depend on Person 4's file."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ----------------------------------------------------------------------------
# Least-squares regression
# ----------------------------------------------------------------------------


SOLVERS = ("qr", "lstsq", "normal")


def design_matrix(t, degree: int, t_center: float = 0.0, t_scale: float = 1.0) -> np.ndarray:
    """Columns [1, tau, tau^2, ..., tau^degree]."""
    tau = (np.asarray(t, dtype=float) - t_center) / t_scale
    return np.vander(tau, degree + 1, increasing=True)


@dataclass
class FitResult:
    degree: int
    coef: np.ndarray        # coefficients in tau (centred, scaled time)
    t_center: float
    t_scale: float
    n: int
    rss: float
    sigma2: float           # residual variance, rss / (n - p)
    cov_unscaled: np.ndarray  # (X^T X)^{-1}
    cond_X: float
    cond_normal: float      # cond(X^T X) = cond(X)^2
    solver: str

    @property
    def n_params(self) -> int:
        return self.degree + 1

    @property
    def aic(self) -> float:
        return self.n * np.log(self.rss / self.n) + 2 * self.n_params

    @property
    def bic(self) -> float:
        return self.n * np.log(self.rss / self.n) + np.log(self.n) * self.n_params

    def design(self, t) -> np.ndarray:
        return design_matrix(t, self.degree, self.t_center, self.t_scale)

    def predict(self, t, level: float | None = None):
        """Fitted values; with ``level`` also a (lo, hi) prediction interval.

        The interval assumes independent errors. Annual temperature residuals
        are autocorrelated, so it is optimistic - backtest coverage in
        ``validation.py`` measures by how much.
        """
        X = self.design(t)
        mean = X @ self.coef
        if level is None:
            return mean
        dof = max(self.n - self.n_params, 1)
        se = np.sqrt(self.sigma2 * (1.0 + np.einsum("ij,jk,ik->i", X, self.cov_unscaled, X)))
        half = stats.t.ppf(0.5 + level / 2, dof) * se
        return mean, mean - half, mean + half

    def slope(self, t) -> np.ndarray:
        """d(anomaly)/d(year) at year t, in deg C per year."""
        tau = (np.atleast_1d(np.asarray(t, float)) - self.t_center) / self.t_scale
        d = np.zeros_like(tau)
        for j in range(1, self.degree + 1):
            d += j * self.coef[j] * tau ** (j - 1)
        return d / self.t_scale

    def equation(self) -> str:
        terms = [f"{c:+.4f}" + ("" if j == 0 else "·τ" + ("" if j == 1 else f"^{j}")) for j, c in enumerate(self.coef)]
        return f"y = {' '.join(terms)}   τ = (year − {self.t_center:g}) / {self.t_scale:g}"


def fit_polynomial(t, y, degree: int = 1, solver: str = "qr", center: bool = True) -> FitResult:
    """Least-squares polynomial fit of the given degree.

    center=False uses raw calendar years in the design matrix; this is only
    useful for demonstrating how badly conditioned that choice is.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}")
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) <= degree + 1:
        raise ValueError("need more observations than parameters")
    t_center, t_scale = ((t.min() + t.max()) / 2, (t.max() - t.min()) / 2) if center else (0.0, 1.0)
    X = design_matrix(t, degree, t_center, t_scale)

    Q, R = qr(X, mode="economic")
    if solver == "qr":
        coef = solve_triangular(R, Q.T @ y)
    elif solver == "lstsq":
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        coef = np.linalg.solve(X.T @ X, X.T @ y)  # textbook formula, squares cond(X)

    rss = float(np.sum((y - X @ coef) ** 2))
    Rinv = solve_triangular(R, np.eye(R.shape[0]))
    return FitResult(degree, coef, t_center, t_scale, len(t), rss, rss / (len(t) - degree - 1),
                     Rinv @ Rinv.T, float(np.linalg.cond(X)), float(np.linalg.cond(X.T @ X)), solver)


def extrapolate(fit: FitResult, years, level: float = 0.95) -> pd.DataFrame:
    """Predictions with prediction interval for future years."""
    years = np.asarray(years, float)
    mean, lo, hi = fit.predict(years, level)
    return pd.DataFrame({"year": years.astype(int), "prediction": mean, "lo": lo, "hi": hi})


def compare_degrees(t, y, degrees=(1, 2, 3)) -> pd.DataFrame:
    """In-sample comparison of polynomial degrees."""
    rows = []
    for d in degrees:
        f = fit_polynomial(t, y, d)
        res = np.asarray(y) - f.predict(t)
        rows.append({"degree": d, "rmse": _rmse(y, f.predict(t)), "mae": _mae(y, f.predict(t)),
                     "r2": 1 - f.rss / np.sum((np.asarray(y) - np.mean(y)) ** 2),
                     "aic": f.aic, "bic": f.bic, "cond_X": f.cond_X,
                     "lag1_autocorr": float(np.corrcoef(res[:-1], res[1:])[0, 1])})
    return pd.DataFrame(rows).set_index("degree")


def split_by_year(t, y, cutoff: int):
    """Train on years <= cutoff, test on years > cutoff."""
    t, y = np.asarray(t), np.asarray(y)
    tr = t <= cutoff
    return t[tr], y[tr], t[~tr], y[~tr]


def conditioning_table(t, y, degrees=(1, 2, 3)) -> pd.DataFrame:
    """Raw years vs. centred time, QR vs. normal equations.

    ``fit_err`` is the max difference between the fitted values of a solver and
    those of the centred-QR reference, in deg C.
    """
    ref = {d: fit_polynomial(t, y, d, "qr", True).predict(t) for d in degrees}
    rows = []
    for d in degrees:
        for center in (False, True):
            for solver in ("normal", "qr"):
                try:
                    f = fit_polynomial(t, y, d, solver, center)
                    err = float(np.max(np.abs(f.predict(t) - ref[d])))
                    cond_x, cond_n = f.cond_X, f.cond_normal
                except np.linalg.LinAlgError:
                    err, cond_x, cond_n = np.inf, np.nan, np.nan
                rows.append({"degree": d, "time": "centred" if center else "raw years", "solver": solver,
                             "cond(X)": cond_x, "cond(XtX)": cond_n, "fit_err": err})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# Different historical periods, same model
# ----------------------------------------------------------------------------


def forecast_spread(t, y, target_year: int = 2040, degrees=(1, 2), starts=(1880, 1900, 1920, 1940, 1960, 1980)) -> pd.DataFrame:
    """Extrapolated value at ``target_year`` from different training-period starts."""
    t, y = np.asarray(t), np.asarray(y)
    rows = []
    for d in degrees:
        for s0 in starts:
            m = t >= s0
            if m.sum() <= d + 3:  # window starts too close to the end of the data
                continue
            f = fit_polynomial(t[m], y[m], d)
            rows.append({"degree": d, "start": s0, "n_train": int(m.sum()),
                         "forecast": float(f.predict([target_year])[0]),
                         "slope_per_decade_at_end": float(f.slope([t[-1]])[0] * 10)})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------


DEG_COLOR = {1: CAT[0], 2: CAT[1], 3: CAT[2]}


DEG_NAME = {1: "linear", 2: "quadratic", 3: "cubic"}


def plot_fits(t, y, fits: list, future_end: int = 2040, level: float = 0.95, bands: bool = True):
    t, y = np.asarray(t, float), np.asarray(y, float)
    fig, ax = new_fig(h=4.6)
    ax.plot(t, y, "o", color=GRAY, ms=3.5, label="observed")
    grid = np.arange(t[0], future_end + 1, 0.5)
    for f in fits:
        c = DEG_COLOR.get(f.degree, CAT[3])
        mean, lo, hi = f.predict(grid, level)
        ax.plot(grid, mean, color=c, label=f"{DEG_NAME.get(f.degree, f.degree)} (RMSE {np.sqrt(f.rss / f.n):.3f})")
        if bands:
            fut = grid > t[-1]
            ax.fill_between(grid[fut], lo[fut], hi[fut], color=c, alpha=0.13, lw=0)
        end = float(f.predict([future_end])[0])
        ax.annotate(f"{end:.2f}", (future_end, end), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                    color=INK_2, va="center")
    ax.axvline(t[-1], color=INK_2, lw=0.8, ls=":")
    ax.set_xlim(t[0] - 2, future_end + 6)
    ax.set_xlabel("year")
    ax.set_ylabel("global anomaly (°C, vs 1951–80)")
    ax.legend(loc="upper left")
    ax.set_title(f"Least-squares trends extrapolated to {future_end} ({int(level * 100)}% prediction bands)")
    return fig


def plot_forecast_spread(df, target_year=2040):
    fig, ax = new_fig(w=6.4, h=3.6)
    for d, g in df.groupby("degree"):
        ax.plot(g["start"], g["forecast"], "-o", color=DEG_COLOR.get(d, CAT[3]), label=DEG_NAME.get(d, d))
    ax.set_xlabel("first year of training window (training ends at last observation)")
    ax.set_ylabel(f"forecast for {target_year} (°C)")
    ax.set_title(f"Same model, different training period: {target_year} forecast")
    ax.legend()
    return fig

# ----------------------------------------------------------------------------
# Run this file
# ----------------------------------------------------------------------------


def main():
    """Person 3's deliverables: fitted curves with 2030/2040 forecasts, and how the answer depends on the training period."""
    df, years, values = load_global()
    fits = [fit_polynomial(years, values, d) for d in (1, 2)]
    save_figure(plot_fits(years, values, fits, 2040), "regression_fits")

    rows = []
    for f in [fit_polynomial(years, values, d) for d in (1, 2, 3)]:
        rows += [{"degree": f.degree, "coef": f"c{k}", "value": c} for k, c in enumerate(f.coef)]
    save_table(pd.DataFrame(rows).pivot(index="degree", columns="coef", values="value"), "regression_coefficients")

    forecasts = []
    for f in fits:
        e = extrapolate(f, [2030, 2040, 2050])
        e.insert(0, "model", {1: "linear", 2: "quadratic"}[f.degree])
        forecasts.append(e)
    forecasts = pd.concat(forecasts, ignore_index=True)
    save_table(forecasts, "regression_extrapolation", index=False)
    save_table(compare_degrees(years, values), "regression_in_sample")
    save_table(conditioning_table(years, values), "regression_conditioning", index=False)

    spread = forecast_spread(years, values)
    save_table(spread, "regression_forecast_spread", index=False)
    save_figure(plot_forecast_spread(spread), "regression_forecast_spread")
    print(forecasts.round(2).to_string(index=False))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# Tests  (python -m pytest person3_regression.py)
# ----------------------------------------------------------------------------


def _line(n=60, noise=0.0, seed=0):
    t = np.arange(1960.0, 1960 + n)
    rng = np.random.default_rng(seed)
    return t, 0.03 * (t - 1960) - 0.2 + noise * rng.normal(size=n)


def test_design_matrix_shape_and_columns():
    X = design_matrix([1, 2, 3], 2)
    assert X.shape == (3, 3) and np.allclose(X[:, 2], [1, 4, 9])


def test_recovers_exact_polynomial():
    t = np.linspace(1900, 2020, 80)
    tau = (t - t.mean()) / ((t.max() - t.min()) / 2)
    y = 0.1 - 0.3 * tau + 0.7 * tau**2
    f = fit_polynomial(t, y, 2)
    assert np.allclose(f.coef, [0.1, -0.3, 0.7]) and f.rss < 1e-20


def test_solvers_agree_when_well_conditioned():
    t, y = _line(noise=0.1)
    c = [fit_polynomial(t, y, 2, s).coef for s in SOLVERS]
    assert np.allclose(c[0], c[1]) and np.allclose(c[0], c[2])


def test_normal_equations_square_the_condition_number():
    t, y = _line(noise=0.1)
    c = fit_polynomial(t, y, 2)                 # centred: well conditioned, identity holds
    assert np.isclose(c.cond_normal, c.cond_X**2, rtol=1e-2) and c.cond_X < 10
    raw = fit_polynomial(t, y, 2, center=False)  # raw years: cond(X^T X) is past 1/eps
    assert raw.cond_X > 1e6 and raw.cond_normal > 1 / np.finfo(float).eps


def test_slope_and_extrapolation():
    t, y = _line()
    f = fit_polynomial(t, y, 1)
    assert np.isclose(f.slope([2000])[0], 0.03)
    assert np.isclose(f.predict([2030])[0], 0.03 * 70 - 0.2)


def test_prediction_interval_widens_away_from_data():
    t, y = _line(noise=0.1)
    f = fit_polynomial(t, y, 1)
    _, lo, hi = f.predict([1990, 2040, 2100], level=0.95)
    w = hi - lo
    assert w[0] < w[1] < w[2]


def test_interval_coverage_for_iid_noise():
    rng = np.random.default_rng(3)
    t = np.arange(100.0); hits = []
    for _ in range(300):
        y = 0.01 * t + rng.normal(0, 0.2, 100)
        f = fit_polynomial(t[:90], y[:90], 1)
        _, lo, hi = f.predict(t[90:], 0.95)
        hits += list((y[90:] >= lo) & (y[90:] <= hi))
    assert 0.92 < np.mean(hits) < 0.98


def test_conditioning_table_shows_normal_equations_degrade():
    t = np.arange(1880.0, 2026); y = 0.007 * (t - 1880) + 0.1 * np.sin(t / 7)
    tab = conditioning_table(t, y, degrees=(3,))
    raw = tab[(tab.time == "raw years") & (tab.solver == "normal")]["fit_err"].iloc[0]
    cen = tab[(tab.time == "centred") & (tab.solver == "normal")]["fit_err"].iloc[0]
    assert raw > 1e3 * cen


def test_forecast_spread_skips_windows_that_start_too_late():
    t = np.arange(1880.0, 1931); y = 0.01 * (t - 1880)
    s = forecast_spread(t, y)  # default starts include 1940+ which have no data
    assert len(s) > 0 and s["start"].max() < 1930

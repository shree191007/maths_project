"""Part 2: polynomial least-squares trends and extrapolation.

Model:  y = X beta + e,  X[i, j] = tau_i ** j,  tau = (year - centre) / scale.

The textbook solution beta = (X^T X)^{-1} X^T y squares the condition number,
so the default solver is QR (X = QR, solve R beta = Q^T y). The normal-equation
solver is kept only so the two can be compared (see ``conditioning_table``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr, solve_triangular

from . import metrics

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
        rows.append({"degree": d, "rmse": metrics.rmse(y, f.predict(t)), "mae": metrics.mae(y, f.predict(t)),
                     "r2": 1 - f.rss / np.sum((np.asarray(y) - np.mean(y)) ** 2),
                     "aic": f.aic, "bic": f.bic, "cond_X": f.cond_X,
                     "lag1_autocorr": float(np.corrcoef(res[:-1], res[1:])[0, 1])})
    return pd.DataFrame(rows).set_index("degree")


# ------------------------------------------------------------------ hold-outs
def split_by_year(t, y, cutoff: int):
    """Train on years <= cutoff, test on years > cutoff."""
    t, y = np.asarray(t), np.asarray(y)
    tr = t <= cutoff
    return t[tr], y[tr], t[~tr], y[~tr]


def cutoff_holdout(t, y, cutoff: int, degrees=(1, 2, 3), level: float = 0.95) -> pd.DataFrame:
    """Train up to ``cutoff``, score on every later year; one row per degree."""
    tr_t, tr_y, te_t, te_y = split_by_year(t, y, cutoff)
    rows = []
    for d in degrees:
        f = fit_polynomial(tr_t, tr_y, d)
        pred, lo, hi = f.predict(te_t, level)
        rows.append({"degree": d, "train_rmse": metrics.rmse(tr_y, f.predict(tr_t)),
                     "test_rmse": metrics.rmse(te_y, pred), "test_mae": metrics.mae(te_y, pred),
                     "test_bias": metrics.bias(te_y, pred)})
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
            rows.append({"trial": trial, "degree": d, "rmse": metrics.rmse(y[hidden], p), "mae": metrics.mae(y[hidden], p)})
    return pd.DataFrame(rows)


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

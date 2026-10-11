"""Part 2 figures."""
from __future__ import annotations

import numpy as np

from . import regression as rg
from .plot_style import CAT, DIV, GRAY, INK_2, SEQ, new_fig

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


def plot_backtest_by_lead(summary):
    """summary: columns degree, lead, rmse, coverage (rg.summarize_backtest(bt, ['degree','lead']))."""
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


def plot_forecast_spread(df, target_year=2040):
    fig, ax = new_fig(w=6.4, h=3.6)
    for d, g in df.groupby("degree"):
        ax.plot(g["start"], g["forecast"], "-o", color=DEG_COLOR.get(d, CAT[3]), label=DEG_NAME.get(d, d))
    ax.set_xlabel("first year of training window (training ends at last observation)")
    ax.set_ylabel(f"forecast for {target_year} (°C)")
    ax.set_title(f"Same model, different training period: {target_year} forecast")
    ax.legend()
    return fig

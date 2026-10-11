"""Part 3 figures."""
from __future__ import annotations

import numpy as np

from . import svd_analysis as sv
from .plot_style import BLUE, CAT, DIV, GRAY, INK_2, ORANGE, new_fig


def plot_scree(s, max_components: int = 8):
    ev = sv.explained_variance(s)[:max_components]
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
    ev = sv.explained_variance(res.s)
    for i in range(n_modes):
        pat = sv.spatial_pattern(res, i)
        axes[i, 0].bar(range(len(names)), pat, color=[BLUE if v >= 0 else ORANGE for v in pat], width=0.65)
        axes[i, 0].axhline(0, color=INK_2, lw=0.8)
        axes[i, 0].set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7.5)
        axes[i, 0].set_ylim(-1.15, 1.15)
        axes[i, 0].set_title(f"Mode {i + 1} pattern ({ev[i]:.1%})")
        axes[i, 1].plot(years, sv.temporal_amplitude(res, i), color=BLUE, lw=1.6)
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

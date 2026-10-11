"""Part 1 figures."""
from __future__ import annotations

import numpy as np

from . import interpolation as ip
from .plot_style import BLUE, CAT, GRAY, INK, INK_2, new_fig

# one colour per method, used in every panel
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
        if name in ip.METHODS:
            a1.plot(grid, ip.METHODS[name](years[known], values[known], grid), lw=1.4, color=METHOD_COLOR[name], label=name)
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

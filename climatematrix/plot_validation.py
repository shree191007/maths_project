"""Part 4 figures."""
from __future__ import annotations

import numpy as np

from .plot_style import BLUE, CAT, GRAY, INK_2, ORANGE, new_fig


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

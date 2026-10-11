"""Shared matplotlib style: validated categorical palette, one-hue sequential,
blue<->red diverging with a gray midpoint, thin marks and a recessive grid."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# categorical slots 1..8 in their fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, GRAY = CAT[0], CAT[1], CAT[2], "#8c8b86"

SEQ = LinearSegmentedColormap.from_list("cm_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("cm_div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a89", "#b3302f"])


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "figure.dpi": 110, "axes.prop_cycle": mpl.cycler(color=CAT),
    })


apply_style()


def new_fig(ncols=1, nrows=1, w=7.5, h=4.2, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols if ncols > 1 else w, h * nrows if nrows > 1 else h),
                           constrained_layout=True, **kw)
    return fig, ax

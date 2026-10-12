"""Person 2: DATA  (the foundation every other file builds on)

What this file does
    1. Downloads NASA GISTEMP v4 (zonal annual temperature anomalies, 1880-2025).
    2. Cleans it: '***' becomes NaN, duplicate years are dropped, rows are sorted.
    3. Builds the region-by-year matrix A (8 latitude bands x 146 years).
    4. Saves the cleaned table to data/processed/climate.csv.
    5. Plots the historical trends.
    6. Holds the things all four files share: the chart style and the helpers that
       save figures / tables into reports/.

Run it
    python person2_data.py            # writes climate.csv and reports/figures/data_*.png
    python -m pytest person2_data.py  # tests for this file

Anomaly = how much warmer (+) or colder (-) a year was than the 1951-1980 average, in deg C.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are saved as PNG or shown through Streamlit, never in a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


# ----------------------------------------------------------------------------
# 1. Loading and cleaning the data
# ----------------------------------------------------------------------------


GISTEMP_ZONAL_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "raw" / "ZonAnn.Ts+dSST.csv"


PROCESSED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "climate.csv"


BANDS = [
    ("64N-90N", 64, 90),
    ("44N-64N", 44, 64),
    ("24N-44N", 24, 44),
    ("EQU-24N", 0, 24),
    ("24S-EQU", -24, 0),
    ("44S-24S", -44, -24),
    ("64S-44S", -64, -44),
    ("90S-64S", -90, -64),
]


BAND_NAMES = [b[0] for b in BANDS]


def download_gistemp(dest: Path = DEFAULT_PATH, url: str = GISTEMP_ZONAL_URL, force: bool = False) -> Path:
    """Download the zonal-means CSV from NASA GISS (about 10 KB)."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_zonal(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the zonal table: index = Year, columns = Glob, NHem, ..., bands."""
    path = Path(path)
    if not path.exists():
        download_gistemp(path)
    df = pd.read_csv(path, na_values=["***", "****"])
    df = df.drop_duplicates("Year").set_index("Year").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def clean_series(s: pd.Series) -> pd.Series:
    """Drop NaNs and sort by year; the result is the 'known observations'."""
    return s.dropna().sort_index()


def global_series(df: pd.DataFrame, start: int | None = None, end: int | None = None) -> pd.Series:
    """Global-mean annual anomaly for [start, end], missing years removed."""
    return clean_series(df["Glob"].loc[start:end])


def region_matrix(df: pd.DataFrame, start: int | None = None, end: int | None = None, bands=BAND_NAMES):
    """Return (A, years, names): A is (regions x years), NaN where missing."""
    sub = df.loc[start:end, list(bands)]
    return sub.to_numpy(dtype=float).T, sub.index.to_numpy(), list(bands)


def band_area_weights(bands=BAND_NAMES) -> np.ndarray:
    """Fraction of the sphere's area in each band: (sin(lat_n) - sin(lat_s)) / 2."""
    edges = {n: (lo, hi) for n, lo, hi in BANDS}
    w = np.array([(np.sin(np.radians(edges[b][1])) - np.sin(np.radians(edges[b][0]))) / 2 for b in bands])
    return w


def quality_report(path: Path = DEFAULT_PATH) -> dict:
    """What the raw file contains *before* cleaning: rows, years, duplicate years,
    missing cells ('***' in the CSV) and the sanity range of the values."""
    raw = pd.read_csv(path, na_values=["***", "****"])
    values = raw.drop(columns="Year").apply(pd.to_numeric, errors="coerce")
    return {
        "rows": len(raw),
        "first_year": int(raw["Year"].min()),
        "last_year": int(raw["Year"].max()),
        "duplicate_years": int(raw["Year"].duplicated().sum()),
        "missing_cells": int(values.isna().sum().sum()),
        "min_anomaly": float(np.nanmin(values.to_numpy())),
        "max_anomaly": float(np.nanmax(values.to_numpy())),
    }


def export_processed(df: pd.DataFrame | None = None, dest: Path = PROCESSED_PATH) -> Path:
    """Write the cleaned table (Year, Glob and the 8 latitude bands) to
    data/processed/climate.csv. Years are rows here; the region-by-year matrix A
    used by the SVD is this table's band columns transposed."""
    df = load_zonal() if df is None else df
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df[["Glob"] + BAND_NAMES].to_csv(dest)
    return dest


def load_global(start: int = 1880, end: int | None = None):
    """(zonal table, years, global-mean anomalies) for the chosen study period."""
    df = load_zonal()
    g = global_series(df, start, end)
    return df, g.index.to_numpy(float), g.to_numpy()


def save_figure(fig, name: str) -> Path:
    """Save a matplotlib figure as reports/figures/<name>.png and close it."""
    path = REPORTS / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, **kw) -> Path:
    """Save a DataFrame as reports/tables/<name>.csv."""
    path = REPORTS / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kw)
    return path


# ----------------------------------------------------------------------------
# 2. Shared chart style (used by every file)
# ----------------------------------------------------------------------------


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
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "figure.dpi": 110, "axes.prop_cycle": matplotlib.cycler(color=CAT),
    })


apply_style()


def new_fig(ncols=1, nrows=1, w=7.5, h=4.2, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols if ncols > 1 else w, h * nrows if nrows > 1 else h),
                           constrained_layout=True, **kw)
    return fig, ax


# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------


def plot_trends(df: pd.DataFrame):
    """Global anomaly (bold) over the eight latitude bands (thin): the 'historical trend' figure."""
    fig, ax = new_fig(h=4.4)
    for i, b in enumerate(BAND_NAMES):
        ax.plot(df.index, df[b], lw=0.9, color=CAT[i], alpha=0.75, label=b)
    ax.plot(df.index, df["Glob"], lw=2.6, color="#0b0b0b", label="Global")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("anomaly (°C vs 1951–1980)")
    ax.set_title("GISTEMP annual temperature anomalies by latitude band")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    return fig


def plot_matrix(A: np.ndarray, years, names):
    """Heat map of the region-by-year matrix A (rows = regions, columns = years)."""
    fig, ax = new_fig(h=3.4)
    lim = np.nanmax(np.abs(A))
    im = ax.imshow(A, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest",
                   extent=[years[0], years[-1], len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    ax.set_xlabel("year")
    ax.set_title("The matrix A: one row per latitude band, one column per year")
    fig.colorbar(im, ax=ax, label="anomaly (°C)")
    return fig

# ----------------------------------------------------------------------------
# 4. Run this file
# ----------------------------------------------------------------------------


def main():
    """Person 2's deliverables: the quality report, climate.csv, and the two data figures."""
    df = load_zonal()
    print("Raw file check:", quality_report())
    print("Saved", export_processed(df).relative_to(ROOT))
    A, years, names = region_matrix(df)
    print(f"Matrix A: {A.shape[0]} regions x {A.shape[1]} years")
    save_figure(plot_trends(df), "data_trends")
    save_figure(plot_matrix(A, years, names), "data_matrix")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# 5. Tests  (python -m pytest person2_data.py)
# ----------------------------------------------------------------------------


def test_real_data_shape_and_bands_do_not_overlap():
    df = load_zonal(); A, yrs, names = region_matrix(df, 1880, 2025)
    assert A.shape == (8, 146) and not np.isnan(A).any()
    assert np.isclose(band_area_weights().sum(), 1)
    # area-weighted mean of the 8 bands ~ NASA's global mean (different gridding: 0.016 C mean gap)
    glob = band_area_weights() @ A
    assert np.mean(np.abs(glob - df.loc[1880:2025, "Glob"].to_numpy())) < 0.03


def test_quality_report_and_processed_export(tmp_path):
    q = quality_report()
    assert q["rows"] == 146 and q["duplicate_years"] == 0 and q["missing_cells"] == 0
    out = export_processed(dest=tmp_path / "climate.csv")
    import pandas as pd
    back = pd.read_csv(out, index_col="Year")
    assert list(back.columns) == ["Glob"] + BAND_NAMES and back.shape == (146, 9)

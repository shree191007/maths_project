"""Shared data layer: NASA GISTEMP v4 zonal annual temperature anomalies.

Source: https://data.giss.nasa.gov/gistemp/data_v4.html
Anomalies are in deg C relative to the 1951-1980 mean. Missing values are
written as '***' in the CSV and loaded as NaN.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GISTEMP_ZONAL_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ZonAnn.Ts+dSST.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "climate.csv"

# The eight non-overlapping latitude bands (name, south edge, north edge).
# The file also holds overlapping aggregates (Glob, NHem, 24N-90N, ...) which
# must NOT go in the same matrix as the bands, or rows would be linearly
# dependent by construction.
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


def plot_trends(df: pd.DataFrame):
    """Global anomaly (bold) over the eight latitude bands (thin): the 'historical trend' figure."""
    from .plot_style import CAT, GRAY, new_fig

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
    from .plot_style import DIV, new_fig

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

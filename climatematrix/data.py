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

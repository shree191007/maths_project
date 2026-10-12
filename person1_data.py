"""Person 1: LOAD, CLEAN AND PLOT the NASA data.

Data: NASA GISTEMP v4, global annual temperature anomaly (1880-2025).
An anomaly is how much warmer (+) or colder (-) a year was than the 1951-1980 average, in deg C.

What this file does
    1. load_data()     reads the NASA file and cleans it (missing values, repeated years, order)
    2. plot_history()  plots the anomaly for every year

Run it:   python person1_data.py
Test it:  python -m pytest person1_data.py
"""
from pathlib import Path
import urllib.request

import matplotlib

matplotlib.use("Agg")  # save pictures to files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "raw" / "ZonAnn.Ts+dSST.csv"
DATA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"
REPORTS = ROOT / "reports"  # all pictures and tables are saved here


def load_data():
    """Return (years, anomalies) as two clean arrays."""
    if not DATA_FILE.exists():  # download once if the file is missing
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    table = pd.read_csv(DATA_FILE, na_values=["***"])  # NASA writes *** for "missing"
    table = table[["Year", "Glob"]]  # Glob = global average anomaly
    table = table.drop_duplicates("Year")  # keep one row per year
    table = table.dropna()  # drop any year with a missing value
    table = table.sort_values("Year")  # put the years in order
    return table["Year"].to_numpy(float), table["Glob"].to_numpy(float)


def plot_history(years, anomalies, path=REPORTS / "history.png"):
    """Plot the temperature anomaly against the year."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(years, anomalies, color="#2a78d6", linewidth=1, marker="o", markersize=3)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (deg C)")
    plt.title("NASA global temperature anomaly, 1880-2025")
    plt.grid(alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    years, anomalies = load_data()
    print(f"Loaded {len(years)} years: {int(years[0])} to {int(years[-1])}")
    print(f"Lowest anomaly {anomalies.min():.2f} deg C in {int(years[anomalies.argmin()])}, "
          f"highest {anomalies.max():.2f} deg C in {int(years[anomalies.argmax()])}")
    plot_history(years, anomalies)
    print("Saved reports/history.png")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- tests
def test_load_data_is_clean():
    years, anomalies = load_data()
    assert len(years) == len(anomalies) == 146
    assert years[0] == 1880 and years[-1] == 2025
    assert not np.isnan(anomalies).any()
    assert (np.diff(years) == 1).all()  # in order, no repeats, no gaps

"""Person 3: INTERPOLATION TEST (estimate years INSIDE the observed period).

Interpolation = estimating a value between years we have data for.
To test it fairly we do this:
    1. pick some years (1890, 1900, ..., 2020) and pretend we never saw them
    2. fit the least squares line using only the OTHER years
    3. use that line to estimate the hidden years
    4. compare each estimate with the real NASA value

Error = estimate - actual   (positive: line was too high, negative: too low)
Person 4 turns these errors into MAE and RMSE.

Run it:   python person3_interpolation.py
Test it:  python -m pytest person3_interpolation.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from person1_data import REPORTS, load_data
from person2_regression import fit_line, predict

SELECTED_YEARS = list(range(1890, 2030, 10))  # 1890, 1900, ..., 2020: all inside 1880-2025


def interpolate_selected(years, anomalies, selected=SELECTED_YEARS):
    """Hide the selected years, fit on the rest, estimate the hidden years.

    Returns a table with columns: year, actual, estimated, error.
    """
    hidden = np.isin(years, selected)  # True for the years we hide
    a, b = fit_line(years[~hidden], anomalies[~hidden])  # fit WITHOUT the hidden years
    estimated = predict(a, b, years[hidden])
    actual = anomalies[hidden]
    return pd.DataFrame({"year": years[hidden].astype(int), "actual": actual,
                         "estimated": estimated, "error": estimated - actual})


def plot_interpolation(years, anomalies, table, path=REPORTS / "interpolation.png"):
    """Plot the data, with each hidden year's estimate on the line."""
    hidden = np.isin(years, table["year"])
    a, b = fit_line(years[~hidden], anomalies[~hidden])
    plt.figure(figsize=(8, 4.5))
    plt.plot(years[~hidden], anomalies[~hidden], "o", color="gray", markersize=3, label="years used to fit the line")
    plt.plot(years, predict(a, b, years), color="#eb6834", linewidth=2, label="fitted line")
    plt.plot(table["year"], table["actual"], "o", mfc="none", mec="black", markersize=8, label="hidden years: actual")
    plt.plot(table["year"], table["estimated"], "x", color="#2a78d6", markersize=8, mew=2, label="hidden years: estimated")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (deg C)")
    plt.title("Interpolation: estimating hidden years from the line")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    years, anomalies = load_data()
    table = interpolate_selected(years, anomalies)
    print(table.round(3).to_string(index=False))
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "interpolation_errors.csv", index=False)
    plot_interpolation(years, anomalies, table)
    print("Saved reports/interpolation_errors.csv and reports/interpolation.png")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- tests
def test_hidden_years_are_not_used_for_fitting():
    # data that is a perfect line, except the hidden years hold a wild value:
    years = np.arange(1880.0, 2026.0)
    y = 0.01 * (years - 1880)
    y_wild = y.copy()
    y_wild[np.isin(years, SELECTED_YEARS)] = 99.0
    table = interpolate_selected(years, y_wild)
    # the line never saw the 99s, so it estimates the true line values
    assert np.allclose(table["estimated"], 0.01 * (table["year"] - 1880))


def test_table_has_one_row_per_selected_year():
    years, anomalies = load_data()
    table = interpolate_selected(years, anomalies)
    assert list(table["year"]) == SELECTED_YEARS
    assert np.allclose(table["error"], table["estimated"] - table["actual"])

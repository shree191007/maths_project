"""Person 4: EXTRAPOLATION + MAE / RMSE + FINAL RESULTS.

Extrapolation = using the line to estimate years OUTSIDE the observed period (the future).

This file:
    1. measures errors with MAE and RMSE
    2. reports the interpolation errors (from Person 3)
    3. extrapolates the fitted line to 2030 and 2040
    4. checks how reliable extrapolation is by testing it on the recent past
    5. draws the final picture

    error = estimate - actual
    MAE   = average of |error|             (the typical size of a miss)
    RMSE  = sqrt( average of error^2 )     (like MAE, but big misses count more)

Run it:   python person4_results.py        (runs the whole project's results)
Test it:  python -m pytest person4_results.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from person1_data import REPORTS, load_data
from person2_regression import fit_line, predict
from person3_interpolation import interpolate_selected

FUTURE_YEARS = [2030, 2040]


def mae(actual, estimated):
    """Mean absolute error."""
    errors = np.asarray(estimated, dtype=float) - np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(errors)))


def rmse(actual, estimated):
    """Root mean square error."""
    errors = np.asarray(estimated, dtype=float) - np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))


def extrapolate(a, b, future_years=FUTURE_YEARS):
    """Estimated anomalies for years beyond the data, from the line y = a + b x."""
    return pd.DataFrame({"year": future_years, "estimated_anomaly": predict(a, b, future_years)})


def check_extrapolation(years, anomalies, cutoff=2005):
    """Pretend it is `cutoff`: fit on years up to then, estimate every later year, compare with the truth."""
    train = years <= cutoff
    a, b = fit_line(years[train], anomalies[train])
    estimated = predict(a, b, years[~train])
    return years[~train], anomalies[~train], estimated


def plot_final(years, anomalies, a, b, future, path=REPORTS / "final_result.png"):
    """History, fitted line extended to 2040, and the 2030 and 2040 estimates."""
    line_years = np.arange(years[0], future["year"].max() + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(years, anomalies, "o", color="gray", markersize=3, label="NASA data")
    plt.plot(line_years, predict(a, b, line_years), color="#eb6834", linewidth=2.5, label="fitted line")
    plt.plot(future["year"], future["estimated_anomaly"], "D", color="#2a78d6", markersize=8, label="extrapolated estimates")
    for year, value in zip(future["year"], future["estimated_anomaly"]):
        plt.annotate(f"{year}: {value:.2f}", (year, value), xytext=(-28, -26), textcoords="offset points", fontsize=9)
    plt.axvline(years[-1], color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (deg C)")
    plt.title("Fitted line and extrapolation to 2030 and 2040")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    years, anomalies = load_data()
    a, b = fit_line(years, anomalies)
    print(f"Fitted line on all {len(years)} years: y = {a:.4f} + {b:.6f} * year   ({10 * b:.3f} deg C per decade)\n")

    # 1. how well does the line fit the data it was fitted to?
    fitted = predict(a, b, years)
    print(f"Fit to all data:     MAE = {mae(anomalies, fitted):.3f}   RMSE = {rmse(anomalies, fitted):.3f}  deg C")

    # 2. interpolation: hidden years estimated from a line fitted without them
    table = interpolate_selected(years, anomalies)
    print(f"Interpolation test:  MAE = {mae(table['actual'], table['estimated']):.3f}   "
          f"RMSE = {rmse(table['actual'], table['estimated']):.3f}  deg C   ({len(table)} hidden years)")

    # 3. extrapolation beyond the data
    future = extrapolate(a, b)
    print("\nExtrapolation (a straight-line extension, NOT an official climate forecast):")
    print(future.round(3).to_string(index=False))

    # 4. is extrapolation reliable? test it on years we do know
    later_years, actual, estimated = check_extrapolation(years, anomalies, cutoff=2005)
    print(f"\nExtrapolation test: line fitted to 1880-2005, estimating {int(later_years[0])}-{int(later_years[-1])}:")
    print(f"                     MAE = {mae(actual, estimated):.3f}   RMSE = {rmse(actual, estimated):.3f}  deg C")
    print(f"                     average error = {np.mean(estimated - actual):+.3f} deg C (negative = line too low)")

    REPORTS.mkdir(exist_ok=True)
    future.to_csv(REPORTS / "extrapolation.csv", index=False)
    plot_final(years, anomalies, a, b, future)
    print("\nSaved reports/final_result.png and reports/extrapolation.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- tests
def test_mae_and_rmse_by_hand():
    actual = [0.0, 1.0, 2.0]
    estimated = [0.0, 2.0, 0.0]  # errors: 0, +1, -2
    assert abs(mae(actual, estimated) - 1.0) < 1e-12  # (0 + 1 + 2) / 3
    assert abs(rmse(actual, estimated) - np.sqrt(5 / 3)) < 1e-12  # sqrt((0 + 1 + 4) / 3)


def test_rmse_is_never_smaller_than_mae():
    rng = np.random.default_rng(0)
    actual, estimated = rng.normal(size=50), rng.normal(size=50)
    assert rmse(actual, estimated) >= mae(actual, estimated)


def test_extrapolation_follows_the_line():
    future = extrapolate(1.0, 0.5, [10, 20])
    assert list(future["estimated_anomaly"]) == [6.0, 11.0]


def test_extrapolation_check_uses_only_later_years():
    years = np.arange(1880.0, 2026.0)
    later, actual, estimated = check_extrapolation(years, 0.01 * (years - 1880), cutoff=2005)
    assert later.min() == 2006 and np.allclose(actual, estimated)  # perfect line: perfect extrapolation

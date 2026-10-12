"""Person 2: LEAST SQUARES LINEAR REGRESSION.

We fit a straight line   y = a + b x   to the data, where x = year and y = anomaly.

The idea: for any line, each year has an error (actual - line). Square the errors and add them up:

        SSE = sum( (y_i - (a + b x_i))^2 )

Least squares picks the a and b that make SSE as small as possible. Calculus gives:

        b = sum( (x_i - mean_x) * (y_i - mean_y) ) / sum( (x_i - mean_x)^2 )
        a = mean_y - b * mean_x

b is the warming per year (slope); a is where the line crosses x = 0.

Run it:   python person2_regression.py
Test it:  python -m pytest person2_regression.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from person1_data import REPORTS, load_data


def fit_line(x, y):
    """Least squares line through the points. Returns (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    a = y_mean - b * x_mean
    return a, b


def predict(a, b, x):
    """The line's estimate at x:  a + b x."""
    return a + b * np.asarray(x, dtype=float)


def sse(x, y, a, b):
    """Sum of squared errors of the line y = a + b x (the number least squares minimises)."""
    return float(np.sum((np.asarray(y) - predict(a, b, x)) ** 2))


def plot_fit(years, anomalies, a, b, path=REPORTS / "fitted_line.png"):
    """Plot the data with the fitted line."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(years, anomalies, "o", color="gray", markersize=3, label="NASA data")
    plt.plot(years, predict(a, b, years), color="#eb6834", linewidth=2.5, label=f"fitted line: y = {a:.2f} + {b:.5f} x")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (deg C)")
    plt.title("Least squares line through the NASA data")
    plt.legend()
    plt.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    years, anomalies = load_data()
    a, b = fit_line(years, anomalies)
    print(f"Fitted line: y = {a:.4f} + {b:.6f} * year")
    print(f"Slope b = {b:.5f} deg C per year = {10 * b:.3f} deg C per decade")
    print(f"SSE = {sse(years, anomalies, a, b):.3f}")
    plot_fit(years, anomalies, a, b)
    print("Saved reports/fitted_line.png")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- tests
def test_recovers_an_exact_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    a, b = fit_line(x, 2 + 3 * x)
    assert abs(a - 2) < 1e-12 and abs(b - 3) < 1e-12


def test_matches_numpy_polyfit():
    years, anomalies = load_data()
    b_numpy, a_numpy = np.polyfit(years, anomalies, 1)
    a, b = fit_line(years, anomalies)
    assert abs(b - b_numpy) < 1e-9 and abs(a - a_numpy) < 1e-6


def test_no_other_line_has_smaller_sse():
    years, anomalies = load_data()
    a, b = fit_line(years, anomalies)
    best = sse(years, anomalies, a, b)
    for da, db in [(0.01, 0), (-0.01, 0), (0, 1e-5), (0, -1e-5)]:
        assert sse(years, anomalies, a + da, b + db) > best

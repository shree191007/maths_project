import numpy as np

from climatematrix import data_loader as data, metrics, validation as va


def test_metrics():
    y, p = np.array([0.0, 1, 2]), np.array([0.0, 2, 0])
    assert np.isclose(metrics.rmse(y, p), np.sqrt(5 / 3))
    assert np.isclose(metrics.mae(y, p), 1.0) and metrics.max_error(y, p) == 2
    assert np.isclose(metrics.bias(y, p), -1 / 3)


def test_real_data_shape_and_bands_do_not_overlap():
    df = data.load_zonal(); A, yrs, names = data.region_matrix(df, 1880, 2025)
    assert A.shape == (8, 146) and not np.isnan(A).any()
    assert np.isclose(data.band_area_weights().sum(), 1)
    # area-weighted mean of the 8 bands ~ NASA's global mean (different gridding: 0.016 C mean gap)
    glob = data.band_area_weights() @ A
    assert np.mean(np.abs(glob - df.loc[1880:2025, "Glob"].to_numpy())) < 0.03


def test_noise_amplification_global_lagrange_is_unstable():
    x = np.arange(1880.0, 2026); y = 0.007 * (x - 1880) + 0.1 * np.sin(x / 5)
    amp = va.noise_amplification(x, y, n_trials=5)["amplification"]
    assert amp["linear"] <= 1.05 and amp["lagrange (global)"] > 100


def test_matrix_benchmark_ranks_models_sensibly_on_low_rank_data():
    rng = np.random.default_rng(0); yrs = np.arange(60.0)
    A = rng.normal(size=(8, 1)) @ np.sin(yrs / 3)[None, :] + 0.02 * rng.normal(size=(8, 60))
    s = va.summarize(va.matrix_reconstruction_benchmark(A, yrs, ranks=(0, 1), n_trials=5)).rmse
    assert s["SVD rank 1"] < s["row mean (rank 0)"] / 3


def test_training_sensitivity_and_spread_shapes():
    t = np.arange(1880.0, 2026); y = 0.007 * (t - 1880) ** 1.3 / 5
    assert len(va.training_period_sensitivity(t, y)) > 10
    assert len(va.forecast_spread(t, y)) == 12


def test_forecast_spread_skips_windows_that_start_too_late():
    t = np.arange(1880.0, 1931); y = 0.01 * (t - 1880)
    s = va.forecast_spread(t, y)  # default starts include 1940+ which have no data
    assert len(s) > 0 and s["start"].max() < 1930


def test_quality_report_and_processed_export(tmp_path):
    q = data.quality_report()
    assert q["rows"] == 146 and q["duplicate_years"] == 0 and q["missing_cells"] == 0
    out = data.export_processed(dest=tmp_path / "climate.csv")
    import pandas as pd
    back = pd.read_csv(out, index_col="Year")
    assert list(back.columns) == ["Glob"] + data.BAND_NAMES and back.shape == (146, 9)

import numpy as np

from climatematrix import regression as rg


def _line(n=60, noise=0.0, seed=0):
    t = np.arange(1960.0, 1960 + n)
    rng = np.random.default_rng(seed)
    return t, 0.03 * (t - 1960) - 0.2 + noise * rng.normal(size=n)


def test_design_matrix_shape_and_columns():
    X = rg.design_matrix([1, 2, 3], 2)
    assert X.shape == (3, 3) and np.allclose(X[:, 2], [1, 4, 9])


def test_recovers_exact_polynomial():
    t = np.linspace(1900, 2020, 80)
    tau = (t - t.mean()) / ((t.max() - t.min()) / 2)
    y = 0.1 - 0.3 * tau + 0.7 * tau**2
    f = rg.fit_polynomial(t, y, 2)
    assert np.allclose(f.coef, [0.1, -0.3, 0.7]) and f.rss < 1e-20


def test_solvers_agree_when_well_conditioned():
    t, y = _line(noise=0.1)
    c = [rg.fit_polynomial(t, y, 2, s).coef for s in rg.SOLVERS]
    assert np.allclose(c[0], c[1]) and np.allclose(c[0], c[2])


def test_normal_equations_square_the_condition_number():
    t, y = _line(noise=0.1)
    c = rg.fit_polynomial(t, y, 2)                 # centred: well conditioned, identity holds
    assert np.isclose(c.cond_normal, c.cond_X**2, rtol=1e-2) and c.cond_X < 10
    raw = rg.fit_polynomial(t, y, 2, center=False)  # raw years: cond(X^T X) is past 1/eps
    assert raw.cond_X > 1e6 and raw.cond_normal > 1 / np.finfo(float).eps


def test_slope_and_extrapolation():
    t, y = _line()
    f = rg.fit_polynomial(t, y, 1)
    assert np.isclose(f.slope([2000])[0], 0.03)
    assert np.isclose(f.predict([2030])[0], 0.03 * 70 - 0.2)


def test_prediction_interval_widens_away_from_data():
    t, y = _line(noise=0.1)
    f = rg.fit_polynomial(t, y, 1)
    _, lo, hi = f.predict([1990, 2040, 2100], level=0.95)
    w = hi - lo
    assert w[0] < w[1] < w[2]


def test_interval_coverage_for_iid_noise():
    rng = np.random.default_rng(3)
    t = np.arange(100.0); hits = []
    for _ in range(300):
        y = 0.01 * t + rng.normal(0, 0.2, 100)
        f = rg.fit_polynomial(t[:90], y[:90], 1)
        _, lo, hi = f.predict(t[90:], 0.95)
        hits += list((y[90:] >= lo) & (y[90:] <= hi))
    assert 0.92 < np.mean(hits) < 0.98


def test_rolling_origin_never_uses_future():
    t, y = _line(80, 0.05)
    bt = rg.rolling_origin(t, y, degrees=(1,), horizon=5, min_train=30)
    assert (bt["year"] > bt["origin"]).all() and bt["lead"].between(1, 5).all()


def test_conditioning_table_shows_normal_equations_degrade():
    t = np.arange(1880.0, 2026); y = 0.007 * (t - 1880) + 0.1 * np.sin(t / 7)
    tab = rg.conditioning_table(t, y, degrees=(3,))
    raw = tab[(tab.time == "raw years") & (tab.solver == "normal")]["fit_err"].iloc[0]
    cen = tab[(tab.time == "centred") & (tab.solver == "normal")]["fit_err"].iloc[0]
    assert raw > 1e3 * cen

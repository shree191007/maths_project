# ClimateMatrix

A linear-algebra approach to climate trend reconstruction and future temperature estimation, built on
NASA GISTEMP v4 zonal annual anomalies (1880–2025, eight latitude bands).

> **Research question.** How effectively can interpolation and low-rank matrix approximation reconstruct
> missing climate data, and how reliable is extrapolation beyond the observed period?

Results and interpretation: [REPORT.md](REPORT.md).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                 # 38 tests
python scripts/run_all.py           # all deliverables -> reports/figures, reports/tables (about 12 s)
python scripts/run_all.py 3         # just one part (1-4)
streamlit run app.py                # interactive dashboard
```

The dataset (`data/raw/ZonAnn.Ts+dSST.csv`, about 10 KB) is already included. `data.download_gistemp(force=True)`
re-fetches it from
<https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv>.

## The four parts

Each part is one algorithm module, one plotting module and one test file. Parts depend only on `data_loader.py`,
`metrics.py` and the contracts below, so they can be developed in parallel.

| Part | Owner focus | Code | Math | Deliverable |
|---|---|---|---|---|
| **1. Data & interpolation** | reconstruct missing years | [data_loader.py](climatematrix/data_loader.py), [interpolation.py](climatematrix/interpolation.py), [plot_interpolation.py](climatematrix/plot_interpolation.py) | Lagrange `P(t)=ΣTᵢLᵢ(t)`, Vandermonde system `Vc=y`, tridiagonal system for natural cubic splines | actual vs reconstructed graph; table of interpolation errors (`reports/tables/p1_*`) |
| **2. Regression & extrapolation** | trend + future years | [regression.py](climatematrix/regression.py), [plot_regression.py](climatematrix/plot_regression.py) | design matrix `X`, least squares via QR (`Rβ=Qᵀy`), normal equations vs conditioning, prediction intervals | history-vs-predicted graph with coefficients, 2030/2040/2050 forecasts, hold-out errors (`p2_*`) |
| **3. SVD & patterns** | dominant regional modes | [svd_analysis.py](climatematrix/svd_analysis.py), [plot_svd.py](climatematrix/plot_svd.py) | `A=UΣVᵀ`, centring, Eckart–Young, low-rank + soft-threshold matrix completion | explained-variance plot; original vs reconstructed (`p3_*`) |
| **4. Validation & dashboard** | compare everything fairly | [validation.py](climatematrix/validation.py), [metrics.py](climatematrix/metrics.py), [plot_validation.py](climatematrix/plot_validation.py), [pipeline.py](climatematrix/pipeline.py), [app.py](app.py) | error norms (ℓ₁, ℓ₂, ℓ∞), condition numbers, noise amplification | dashboard, comparison tables (`p4_*`), report |

```
NASA GISTEMP ──► data_loader.py ─┬─► Part 1 interpolation ──┐
                          ├─► Part 2 regression ─────┼─► Part 4 validation ─► dashboard + report
                          └─► Part 3 SVD ────────────┘
```

### Interface contracts (what Part 4 relies on)

| Part | Function signatures that must stay stable |
|---|---|
| 1 | `METHODS: dict[str, f(x_known, y_known, x_query) -> ndarray]`; `holdout_experiment(years, values, ...) -> DataFrame[trial, method, rmse, mae]`; `make_hidden_mask` |
| 2 | `fit_polynomial(t, y, degree) -> FitResult` with `.predict(t, level)`, `.slope`, `.cond_X`; `rolling_origin(...) -> DataFrame[degree, origin, lead, year, y_true, y_pred, lo, hi]` |
| 3 | `compute_svd(A, center_mode, weights) -> SVDResult`; `reconstruct(res, k)`; `impute_lowrank(A_with_NaN, k)`; `impute_soft(A_with_NaN, lam)` |
| 4 | `metrics.rmse / mae / max_error / bias`; every benchmark returns a DataFrame |

Where each task in the brief lives:

- Download and clean GISTEMP → `data.download_gistemp`, `data.load_zonal` (`***` → NaN), `clean_series`
- Linear, Lagrange, cubic spline; hide and test → `interpolation.py` (`holdout_experiment`, `runge_experiment`)
- Linear and quadratic (and cubic) regression, design matrix, least squares, 2030/2040 → `regression.py`
- Hold-out periods → `cutoff_holdout`, `rolling_origin`, `random_holdout`
- Matrix, centring, SVD, rank-k reconstruction, error vs rank → `svd_analysis.py` (`rank_table`)
- RMSE/MAE, rank evaluation, training-period sensitivity → `validation.py`
- Streamlit dashboard → `app.py`

## Layout

```
app.py                  Streamlit dashboard (5 tabs)
climatematrix/          the package (modules above, plus plot_style.py shared chart style)
scripts/run_all.py      regenerate every figure and table
tests/                  pytest suite (one file per part)
data/raw/               GISTEMP CSV
reports/figures|tables  generated outputs
```

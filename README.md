# ClimateMatrix

A linear-algebra approach to climate trend reconstruction and future temperature estimation, built on
NASA GISTEMP v4 zonal annual anomalies (1880–2025, eight latitude bands).

> **Research question.** How effectively can interpolation and low-rank matrix approximation reconstruct
> missing climate data, and how reliable is extrapolation beyond the observed period?

The whole project is **four Python files, one per person**. Results and interpretation are in
[REPORT.md](REPORT.md); who did what, in simple words, is in [WHO_DID_WHAT.md](WHO_DID_WHAT.md); every function in
every file is documented in [DOCUMENTATION.md](DOCUMENTATION.md); the speaking script is
[PRESENTATION_SCRIPT.md](PRESENTATION_SCRIPT.md).

## Quick start

```bash
pip install -r requirements.txt
python person4_validation.py all     # runs all four files; writes every figure and table (about 12 s)
streamlit run person4_validation.py  # the interactive dashboard
pytest                               # 39 tests, spread across the four files
```

Each file also runs on its own:

```bash
python person2_data.py        # climate.csv + data figures
python person1_math.py        # interpolation + SVD results
python person3_regression.py  # trend fits + 2030/2040 forecasts
python person4_validation.py  # all the error tests (add `all` to run the other three first)
```

The dataset (`data/raw/ZonAnn.Ts+dSST.csv`, about 10 KB) is included. `download_gistemp(force=True)` in
`person2_data.py` re-fetches it from <https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv>.

## The four files

| File | Person | What is inside |
|---|---|---|
| [person2_data.py](person2_data.py) | **2 · Data** | download and clean GISTEMP (`load_zonal`), build the region × year matrix (`region_matrix`), save `data/processed/climate.csv`, trend plots, **plus the chart style and the save helpers every file uses** |
| [person1_math.py](person1_math.py) | **1 · Mathematics** | interpolation: `linear_interpolate`, `lagrange_interpolate`, `cubic_spline_interpolate`, `vandermonde_interpolate`; SVD: `compute_svd`, `explained_variance`, `reconstruct`, `rank_table`, `impute_lowrank`, `impute_soft` |
| [person3_regression.py](person3_regression.py) | **3 · Regression** | `design_matrix`, `fit_polynomial` (least squares by QR), `FitResult.predict` with intervals, `extrapolate`, `conditioning_table`, `forecast_spread` |
| [person4_validation.py](person4_validation.py) | **4 · Validation + dashboard** | `rmse`, `mae`; hidden-data tests for interpolation, regression and SVD; training-period and noise sensitivity; the Streamlit dashboard; the "run everything" entry point |

Each file contains its own plots and its own tests (`pytest person1_math.py`, and so on), and its docstring at the top
says what it does and how to run it.

```
person2_data.py  ──►  person1_math.py       ──┐
   (data, style,   ──►  person3_regression.py ─┼──►  person4_validation.py
    save helpers)                               ┘       (tests everything + dashboard)
```

The arrows are `import`s and never go backwards, so each person can work on their own file.
`person1_math.py` and `person3_regression.py` do not import each other.

### What each file promises the others

| File | Used by others |
|---|---|
| `person2_data.py` | `load_zonal()`, `load_global()`, `region_matrix(df)` → `(A, years, names)`, `band_area_weights()`, chart style (`new_fig`, `CAT`, `BLUE`, …), `save_figure`, `save_table` |
| `person1_math.py` | `METHODS` (name → `f(x_known, y_known, x_query)`), `make_hidden_mask`, `predict_hidden`, `compute_svd`, `rank_table`, `random_mask`, `block_mask`, `impute_lowrank`, `impute_soft` |
| `person3_regression.py` | `fit_polynomial(t, y, degree)` → `FitResult` (`.predict(t, level)`, `.slope`, `.coef`), `extrapolate`, `forecast_spread` |
| `person4_validation.py` | nothing: it is the top of the chain |

Where each task from the brief lives:

- Download and clean GISTEMP, region × year matrix, trend plot, `climate.csv` → `person2_data.py`
- Linear and Lagrange interpolation, cubic spline → `person1_math.py` (Part A)
- Matrix, `np.linalg.svd`, rebuild with fewer singular values, reconstruction error → `person1_math.py` (Part B)
- Linear and quadratic regression, least squares, 2030 / 2040, different historical periods → `person3_regression.py`
- MAE and RMSE; test interpolation on hidden data; test regression on later years; compare SVD errors; dashboard → `person4_validation.py`

## Layout

```
person1_math.py  person2_data.py  person3_regression.py  person4_validation.py   the whole project
data/raw/            GISTEMP CSV                        data/processed/climate.csv   the cleaned table
reports/figures|tables   everything the four files generate
requirements.txt  pytest.ini  .streamlit/config.toml     setup
```

Output names say where they come from: `data_*`, `interpolation_*`, `svd_*`, `regression_*` are produced by one
person's file; `validation_*` come from Person 4's tests.

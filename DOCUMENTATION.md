# ClimateMatrix: full code documentation

This is the reference for everything in the four code files: what each file is for, every function, class and constant,
what goes in, what comes out, how it works, the tests, and the files each person's code produces.

- **[1. Overview](#1-overview)**: how the four files fit together
- **[2. Person 2: Data](#2-person-2-data--person2_datapy)** · `person2_data.py`
- **[3. Person 1: Mathematics](#3-person-1-mathematics--person1_mathpy)** · `person1_math.py`
- **[4. Person 3: Regression](#4-person-3-regression--person3_regressionpy)** · `person3_regression.py`
- **[5. Person 4: Validation + dashboard](#5-person-4-validation--dashboard--person4_validationpy)** · `person4_validation.py`
- **[6. Output catalogue](#6-output-catalogue)**: which person's code creates which figure and table
- **[7. Running, testing and extending](#7-running-testing-and-extending)**
- **[8. Limitations](#8-limitations)**

---

## 1. Overview

### The question

> How effectively can interpolation and low-rank matrix approximation reconstruct missing climate data, and how reliable is
> extrapolation beyond the observed period?

- **Reconstruction** estimates values *inside* the observed years (filling gaps).
- **Extrapolation** estimates values *outside* them (the future).

They are different tasks and are measured separately.

### The data

NASA GISTEMP v4, zonal annual table: one temperature **anomaly** (°C relative to the 1951–1980 mean) per latitude band per year,
1880–2025. The file has 14 data columns: `Glob`, `NHem`, `SHem`, three overlapping aggregates (`24N-90N`, `24S-24N`, `90S-24S`) and eight
**non-overlapping bands**. The matrix work uses only the eight bands:

```
64N-90N   44N-64N   24N-44N   EQU-24N   24S-EQU   44S-24S   64S-44S   90S-64S
```

The real table has **no missing values**, so every "missing" value in this project is hidden on purpose and scored against the known truth.

### How the files fit together

```
person2_data.py  ──►  person1_math.py       ──┐
 (data, chart style,  ──►  person3_regression.py ─┼──►  person4_validation.py
  save helpers)                                    ┘     (error scores, all tests, dashboard)
```

Imports only go forward. `person1_math.py` and `person3_regression.py` do not import each other, and neither imports Person 4's file
(each keeps a private `_rmse`, and Person 3 a `_mae`, for that reason).

### Conventions used throughout

| Thing | Convention |
|---|---|
| Years | NumPy float arrays, e.g. `np.array([1880., 1881., ...])` |
| Temperatures | anomalies in °C |
| Region-by-year matrix `A` | shape `(regions, years)` = `(8, 146)`: **rows are regions, columns are years** |
| Missing entries | `np.nan` |
| Hidden-entry masks | boolean arrays, **`True` = hidden** |
| Tables returned | `pandas.DataFrame` |
| Random experiments | take `seed`/`rng` so results are repeatable |
| Error scores | RMSE and MAE in °C, smaller is better |

### Test counts

| File | Tests |
|---|---|
| `person2_data.py` | 2 |
| `person1_math.py` | 22 |
| `person3_regression.py` | 9 |
| `person4_validation.py` | 6 |
| **Total** | **39** |

---

## 2. Person 2: Data · `person2_data.py`

**Role:** get the NASA data, clean it, build the matrix, plot it, and provide the shared tools (chart style, save helpers) that all
other files use. It is the foundation: it imports none of the other three files.

### 2.1 Constants

| Name | Value / meaning |
|---|---|
| `ROOT` | folder containing the file (the project root) |
| `REPORTS` | `ROOT / "reports"`, where figures and tables are written |
| `GISTEMP_ZONAL_URL` | `https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv` |
| `DEFAULT_PATH` | `data/raw/ZonAnn.Ts+dSST.csv`, the raw download |
| `PROCESSED_PATH` | `data/processed/climate.csv`, the cleaned output |
| `BANDS` | list of `(name, south_edge_deg, north_edge_deg)` for the 8 non-overlapping bands, north to south |
| `BAND_NAMES` | just the 8 names, in the same order (these are the matrix rows) |
| `SURFACE`, `INK`, `INK_2`, `GRID` | chart background, main text, secondary text, gridline colours |
| `CAT` | the 8 categorical colours in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red |
| `BLUE`, `ORANGE`, `AQUA`, `GRAY` | named colours from the palette |
| `SEQ` | one-hue (blue) light-to-dark colour map for magnitudes |
| `DIV` | blue ↔ gray ↔ red colour map for values above and below zero |

### 2.2 Loading and cleaning

**`download_gistemp(dest=DEFAULT_PATH, url=GISTEMP_ZONAL_URL, force=False) -> Path`**
Downloads the CSV (about 10 KB) from NASA. If the file already exists it does nothing unless `force=True`. Creates folders as needed and returns the path.

**`load_zonal(path=DEFAULT_PATH) -> DataFrame`**
The main loader. Downloads the file if it is missing, then cleans it:
1. `na_values=["***", "****"]` turns NASA's missing-value stars into `NaN`;
2. `drop_duplicates("Year")` keeps one row per year;
3. `set_index("Year").sort_index()` makes Year the row label, in order;
4. `to_numeric(errors="coerce")` forces numbers, unparseable cells become `NaN`.

Returns a DataFrame with index `Year` (1880–2025) and 14 columns.

**`clean_series(s) -> Series`**
Drops `NaN`s and sorts by year: the "known observations" of a series.

**`global_series(df, start=None, end=None) -> Series`**
The global-mean anomaly (`Glob` column) for years `start`..`end`, cleaned with `clean_series`.

**`region_matrix(df, start=None, end=None, bands=BAND_NAMES) -> (A, years, names)`**
Builds the region × year matrix. Selects the years and the band columns, then **transposes** (`.T`) so rows are regions and columns are years. Returns the matrix `A` (shape `(8, 146)` for the full record), the `years` array and the list of region `names`. `NaN` stays `NaN`.

**`band_area_weights(bands=BAND_NAMES) -> ndarray`**
Fraction of the Earth's surface in each band: `(sin(north) − sin(south)) / 2`. The eight values sum to 1. Used for optional area weighting in the SVD.

**`quality_report(path=DEFAULT_PATH) -> dict`**
Inspects the **raw** file before cleaning. Keys: `rows`, `first_year`, `last_year`, `duplicate_years`, `missing_cells`, `min_anomaly`, `max_anomaly`.
For the real file: 146 rows, 1880–2025, 0 duplicates, 0 missing cells, anomalies from −2.59 to +3.24 °C.

**`export_processed(df=None, dest=PROCESSED_PATH) -> Path`**
Writes the cleaned table (`Glob` plus the eight bands, years as rows) to `data/processed/climate.csv`. The SVD matrix `A` is the band columns of this table transposed.

**`load_global(start=1880, end=None) -> (df, years, values)`**
Convenience used by every `main()`: the full zonal table, plus the global-mean series for the chosen period as two arrays (`years` float, `values` anomalies).

### 2.3 Shared helpers

**`save_figure(fig, name) -> Path`**: saves a Matplotlib figure as `reports/figures/<name>.png` (150 dpi), creating folders, and closes the figure.

**`save_table(df, name, **kw) -> Path`**: saves a DataFrame as `reports/tables/<name>.csv`; extra keyword arguments (such as `index=False`) go to `to_csv`.

### 2.4 Chart style (used by all four files)

**`apply_style()`**: sets Matplotlib's global look: light surface, thin 2-px lines, recessive grid, no top/right borders, the categorical colour cycle. Called once when the file is imported.

**`new_fig(ncols=1, nrows=1, w=7.5, h=4.2, **kw) -> (fig, ax)`**: creates a figure with `constrained_layout` so labels don't collide. For several columns/rows the size scales with the count; extra keywords (e.g. `sharex`, `gridspec_kw`) go to `plt.subplots`.

### 2.5 Plots

**`plot_trends(df) -> Figure`**: the historical-trend figure: the global anomaly as a thick black line over eight thin coloured lines (one per band), with a zero line and a legend.

**`plot_matrix(A, years, names) -> Figure`**: heat map of `A` (rows = bands, columns = years) on the diverging blue/red colour map, centred on zero, with a colour bar.

### 2.6 Entry point

**`main()`**: prints the quality report, writes `climate.csv`, prints the matrix shape, saves `data_trends.png` and `data_matrix.png`. Run with `python person2_data.py`.

### 2.7 Tests

| Test | What it checks |
|---|---|
| `test_real_data_shape_and_bands_do_not_overlap` | the matrix is `(8, 146)` with no `NaN`; area weights sum to 1; the area-weighted mean of the eight bands is within 0.03 °C (on average) of NASA's own global mean |
| `test_quality_report_and_processed_export` | the report says 146 rows, 0 duplicates, 0 missing; the exported CSV has 146 rows and columns `Glob` + the 8 bands |

### 2.8 Produces

`data/processed/climate.csv`, `reports/figures/data_trends.png`, `reports/figures/data_matrix.png`.

---

## 3. Person 1: Mathematics · `person1_math.py`

**Role:** two mathematical tools. **Part A** estimates a missing year from the known years around it (interpolation). **Part B** treats the data as a matrix and breaks it into patterns (SVD), rebuilds it from fewer patterns, and can fill missing matrix entries.
Imports from `person2_data.py` only (chart style, loaders, save helpers).

### 3.1 Part A: Interpolation

All interpolators have the same signature, `f(x_known, y_known, x_query) -> ndarray`, so they can be swapped freely.

**`_check(x_known, y_known, x_query=None)`** *(private)*
Validates and converts inputs: both are 1-D float arrays of equal length, at least two points, `x_known` **strictly increasing**; and, if queries are given, they must lie **inside** `[x_known[0], x_known[-1]]` (interpolation never extrapolates; otherwise `ValueError`). Returns `(x, y, q)`.

**`linear_interpolate(x_known, y_known, x_query)`**
Piecewise-linear. For each query year `q`, finds the neighbouring known years `x0 < q < x1` (via `np.searchsorted`) and returns
`y0 + (y1 − y0)·(q − x0)/(x1 − x0)`.

**`_lagrange_eval(xn, yn, q)`** *(private)*
The Lagrange formula itself: `P(q) = Σᵢ yᵢ·Lᵢ(q)`, with basis polynomial `Lᵢ(q) = Π_{j≠i} (q − xⱼ)/(xᵢ − xⱼ)`. `Lᵢ` is 1 at node `i` and 0 at every other node, so the polynomial passes exactly through all given points.

**`lagrange_interpolate(x_known, y_known, x_query, window=None)`**
Lagrange interpolation in two modes. Years are first mapped to `[-1, 1]` to keep the products numerically sane.
- `window=None` → **global**: one polynomial of degree n−1 through *all* known points. Fine for ~12 points, numerically explosive for many (Runge phenomenon).
- `window=k` → **local**: for each query year, use only the `k` nearest known nodes (bracketing the query). `window=2` reproduces linear interpolation; `window=4` is exact for cubics.

**`vandermonde_matrix(x, scale=True)`**
Builds `V` with `V[i, j] = xᵢʲ` (columns increasing powers), after mapping nodes to `[-1, 1]` if `scale=True`. This is the matrix form of "find the polynomial through these points".

**`vandermonde_interpolate(x_known, y_known, x_query)`**
Solves the linear system `V·c = y` for the polynomial's monomial coefficients with `np.linalg.solve`, then evaluates. In exact arithmetic this equals global Lagrange; numerically it is far worse. Kept to show why.

**`vandermonde_condition(x)`**
Condition number of `V` (`np.linalg.cond`). It grows exponentially with the number of nodes: about 2×10¹ at 5 nodes, 6×10¹⁷ at 40, beyond what double precision can resolve (about 4.5×10¹⁵).

**`natural_spline_second_derivatives(x, y)`**
The linear-algebra core of the spline. With spacings `hᵢ = xᵢ₊₁ − xᵢ`, the second derivatives `Mᵢ` satisfy the **tridiagonal system**
`hᵢ₋₁Mᵢ₋₁ + 2(hᵢ₋₁+hᵢ)Mᵢ + hᵢMᵢ₊₁ = 6[(yᵢ₊₁−yᵢ)/hᵢ − (yᵢ−yᵢ₋₁)/hᵢ₋₁]`
with `M₀ = Mₙ = 0` ("natural" ends). Solved with `scipy.linalg.solve_banded`. Fewer than 3 points → all zeros.

**`cubic_spline_interpolate(x_known, y_known, x_query)`**
Natural cubic spline: a smooth (continuous first and second derivative) piecewise cubic through every known point, built from the `Mᵢ` above. With fewer than 3 points it falls back to linear. Written from scratch and tested against `scipy.interpolate.CubicSpline(bc_type="natural")`.

**`METHODS`** *(dict)*: the interpolation methods compared everywhere:
`"linear"`, `"lagrange (4-pt local)"`, `"lagrange (6-pt local)"`, `"cubic spline"`. Adding a method here adds it to every test, plot and the dashboard.

**`make_hidden_mask(n, n_hidden, gap_length, rng) -> bool array`**
Chooses which of `n` positions to hide, as runs of `gap_length` consecutive points; the number of runs is `max(1, round(n_hidden / gap_length))`. Rules: the **first and last points are never hidden** (interpolation can't extrapolate), runs never touch (at least one known point between them), placement is random from `rng` (up to 2000 attempts). `True` = hidden.

**`predict_hidden(years, values, hidden, methods=METHODS) -> {name: predictions}`**
Rebuilds only the hidden years from the known ones, once per method.

**`runge_experiment(years, values, node_counts=(5, 8, 12, 16, 20, 30, 40)) -> DataFrame`**
Shows global Lagrange failing. For each node count it picks that many **equispaced** years as nodes, interpolates all other years, and records `nodes`, `cond_vandermonde`, `rmse_lagrange_global`, `rmse_linear`, `rmse_spline`. Node counts that would use every year (leaving nothing to test) are skipped, which matters for short study periods.
Real result: global Lagrange RMSE 0.15 °C at 12 nodes, 7.7 °C at 16, 5×10⁶ °C at 40; linear and spline stay near 0.11 °C.

### 3.2 Part B: SVD of the region × year matrix

The idea: after subtracting each region's mean, `A_c = U Σ Vᵀ`.
- columns of **U**: *spatial patterns* (how regions move together);
- rows of **Vᵀ**: *time series* of those patterns;
- **Σ** (singular values `s`, largest first): how important each pattern is.

The best rank-`k` approximation keeps the first `k` terms; its error is exactly `√(Σ_{i>k} sᵢ²)` (**Eckart–Young theorem**).

**`CENTER_MODES = ("row", "col", "none")`**: allowed centring choices.

**`class SVDResult`**: container returned by `compute_svd`. Fields: `U`, `s`, `Vt`, `mean` (what was subtracted), `weights` (row weights used, ones if unweighted). Property `rank` = number of singular values.

**`center(A, mode="row") -> (A_centred, mean)`**
Removes the baseline so the SVD describes *variation*: `"row"` subtracts each region's own time-mean, `"col"` each year's cross-region mean, `"none"` nothing.

**`compute_svd(A, center_mode="row", weights=None) -> SVDResult`**
The main call: centre, optionally multiply each row by `weights` (e.g. `√band area` so small polar bands don't dominate), then `np.linalg.svd(..., full_matrices=False)`. Raises `ValueError` if `A` contains `NaN` (use the `impute_*` functions for incomplete matrices).

**`explained_variance(s) -> ndarray`**
`sᵢ² / Σ sⱼ²`: the share of the total variance carried by each pattern. Real result: 71.7 %, 19.8 %, 4.7 %, 1.6 %, 1.0 %, …

**`reconstruct(res, k) -> ndarray`**
Rebuilds the full matrix from the first `k` patterns: `(U[:, :k] · s[:k]) @ Vt[:k]`, undoes the row weights, adds the means back. Same shape as the original `A`.

**`spatial_pattern(res, i)`**: pattern `i` over regions in physical units (row weights undone), scaled so its largest magnitude is 1, sign fixed so the pattern sums to ≥ 0 (the SVD's sign is arbitrary).

**`temporal_amplitude(res, i)`**: the matching time series, with the same sign convention and scale, so `pattern × series` rebuilds the mode.

**`rank_table(A, res, ks=None) -> DataFrame`**
Reconstruction error versus rank, index `k` (0 up to full rank; `k = 0` means "region means only"). Columns: `explained_cum` (cumulative variance explained), `frob_error` (measured `‖A − A_k‖_F`), `eckart_young` (theory: `√Σ_{i>k} sᵢ²`), `rel_error` (`frob_error / ‖A − mean‖`), `rmse`, `mae`. For the unweighted fit `frob_error` and `eckart_young` agree to machine precision.
Real result: rank 1 → RMSE 0.313 °C, rank 2 → 0.171, rank 3 → 0.114, rank 8 → 0.

**`numerical_rank(s, tol=None) -> int`**: number of singular values above `tol` (default `eps · s_max · len(s)`): the rank a computer can actually distinguish.

**`random_mask(shape, frac, rng)`**: boolean mask hiding each entry independently with probability `frac`; the first and last **columns** are never hidden (fair comparison with interpolation).

**`block_mask(shape, gap_length, n_gaps_per_region, rng)`**: each region loses `n_gaps_per_region` runs of `gap_length` consecutive years (non-touching, interior).

**`_complete(A_obs, truncate, max_iter, tol)`** *(private)*
The shared matrix-completion loop: fill gaps with each row's mean; then repeat: re-centre by row, replace the matrix by `truncate(...)`, **write the result only into the missing entries**, stop when the relative change is below `tol`. Raises `ValueError` if a region has no observations at all. Returns `(filled matrix, history of changes)`.

**`impute_lowrank(A_obs, k, max_iter=500, tol=1e-8, return_history=False)`**
Hard-rank completion: `truncate` keeps the first `k` singular values (`k = 0` is the plain row-mean fill, the baseline). Recovers an exactly low-rank matrix perfectly (tested), but with only 8 regions **ranks ≥ 3 overfit** (a year with several gaps leaves fewer observations than free parameters).

**`impute_soft(A_obs, lam, max_iter=500, tol=1e-8, return_history=False)`**
Soft-threshold completion: shrinks every singular value, `s → max(s − λ, 0)`, instead of cutting at a hard rank. Smoother and far less prone to overfitting; `λ` is in units of the singular values; a huge `λ` collapses to the row-mean fill.

### 3.3 Plots

| Function | Shows |
|---|---|
| `METHOD_COLOR` *(dict)* | one fixed colour per interpolation method, used by every panel |
| `plot_reconstruction(years, values, hidden, preds, curves=("linear", "cubic spline"))` | top: known points, hidden truth (rings) and interpolant curves; bottom: error at each hidden year per method |
| `plot_runge(df)` | left: error vs nodes (log scale) for global Lagrange / linear / spline; right: Vandermonde condition number with the 1/ε line |
| `plot_scree(s, max_components=8)` | bars of explained variance + cumulative line |
| `plot_modes(res, names, years, n_modes=3)` | for each mode: spatial pattern as bars over regions, and its time series |
| `plot_original_vs_reconstructed(A, Ak, years, names, k)` | three heat maps on one scale: original, rank-`k` reconstruction, residual |
| `plot_error_vs_rank(rank_df, held_out=None)` | in-sample RMSE vs rank, optionally with held-out RMSE |

### 3.4 Entry point

**`main()`**: (A) hides 20 % of the global series (seed 7), rebuilds it, saves the reconstruction figure and the Runge table/figure; (B) builds `A`, runs `compute_svd`, saves the variance and rank tables, scree plot, modes plot and the rank-2 reconstruction figure, and prints the rank table.

### 3.5 Tests (22)

**Interpolation (11):** `test_linear_matches_numpy` · `test_global_lagrange_reproduces_polynomial_exactly` · `test_local_lagrange_window2_is_linear_and_window4_exact_for_cubics` · `test_lagrange_hits_nodes` · `test_vandermonde_agrees_with_lagrange_for_few_nodes` · `test_vandermonde_condition_grows_fast` · `test_natural_spline_matches_scipy` · `test_spline_has_zero_end_curvature` · `test_no_extrapolation_and_input_validation` · `test_hidden_mask_properties` (edges kept, runs of the right length, separated) · `test_runge_experiment_skips_node_counts_larger_than_series`.

**SVD (11):** `test_full_rank_reconstruction_is_exact` · `test_eckart_young_error_equals_tail_singular_values` · `test_error_decreases_monotonically_with_rank` · `test_explained_variance_sums_to_one_and_sorted` · `test_exact_low_rank_has_numerical_rank` · `test_centering_modes` · `test_weighted_svd_round_trips` · `test_hard_impute_recovers_exact_low_rank_matrix` (error < 1e-4) · `test_impute_keeps_observed_entries_and_k0_is_row_mean` · `test_soft_impute_with_large_lambda_collapses_to_row_mean` · `test_masks_spare_edges`. (`_A` is the test helper that builds a random low-rank matrix.)

### 3.6 Produces

`interpolation_reconstruction.png`, `interpolation_runge.png` / `.csv`, `svd_variance.csv`, `svd_rank_table.csv`, `svd_scree.png`, `svd_modes.png`, `svd_reconstruction_rank2.png`.

---

## 4. Person 3: Regression · `person3_regression.py`

**Role:** fit trends to the history by **least squares**, extrapolate to future years with uncertainty, and show how much the answer depends on the training period. Imports from `person2_data.py` only.

### 4.1 The model

`y = X β + e`, where row `i` of the **design matrix** `X` is `[1, τᵢ, τᵢ², …, τᵢᵈ]` and `τ = (year − centre) / scale` (years mapped to about −1…1).
Least squares chooses `β` to minimise `‖Xβ − y‖²`. The textbook answer `β = (XᵀX)⁻¹Xᵀy` squares the condition number, so the default solver is **QR**: `X = QR`, then solve the triangular system `Rβ = Qᵀy`.

### 4.2 Constants and private helpers

| Name | Meaning |
|---|---|
| `SOLVERS = ("qr", "lstsq", "normal")` | allowed solvers: QR (default, stable), `np.linalg.lstsq` (SVD-based), normal equations (textbook, for comparison only) |
| `_rmse`, `_mae` | private copies of the error formulas so this file doesn't need Person 4's |
| `DEG_COLOR`, `DEG_NAME` | colour and name per degree: 1 = linear, 2 = quadratic, 3 = cubic |

### 4.3 Functions and classes

**`design_matrix(t, degree, t_center=0.0, t_scale=1.0) -> ndarray`**
Shifts/scales `t` and builds the columns `[1, τ, …, τᵈ]` with `np.vander(..., increasing=True)`. Shape `(n_years, degree + 1)`.

**`class FitResult`**: everything about a fitted model.
Fields: `degree`, `coef` (coefficients **in τ**), `t_center`, `t_scale`, `n` (observations), `rss` (residual sum of squares), `sigma2` (residual variance `rss/(n − p)`), `cov_unscaled` (`(XᵀX)⁻¹`), `cond_X`, `cond_normal` (`cond(XᵀX)`), `solver`.
- `n_params` → `degree + 1`
- `aic`, `bic` → `n·ln(rss/n) + 2p` and `n·ln(rss/n) + p·ln n` (lower is better; BIC penalises extra terms more)
- `design(t)` → the design matrix for new years with this model's centring
- `predict(t, level=None)` → fitted values `X @ coef`. With `level` (e.g. 0.95) returns `(mean, lo, hi)`: a **prediction interval** `mean ± t_crit · √(σ²(1 + x(XᵀX)⁻¹xᵀ))`, which widens away from the data. *Assumes independent errors, so it is too narrow for annual temperatures (see backtest coverage).*
- `slope(t)` → warming rate d(anomaly)/d(year) in °C per year at year(s) `t`
- `equation()` → readable string such as `y = −0.0860 +0.5990·τ +0.5010·τ^2   τ = (year − 1952.5) / 72.5`

**`fit_polynomial(t, y, degree=1, solver="qr", center=True) -> FitResult`**
The main fit. Raises `ValueError` if there are not more observations than parameters. `center=True` maps years to −1…1 (the recommended, well-conditioned choice); `center=False` uses raw calendar years, only useful to demonstrate how badly conditioned that is. Always computes the QR factorisation to get a stable covariance and the condition numbers.

**`extrapolate(fit, years, level=0.95) -> DataFrame`**
Columns `year`, `prediction`, `lo`, `hi` for the requested (future) years.
Real result, fit to 1880–2025: linear **0.72 °C (2030), 0.81 °C (2040)**; quadratic **1.13 °C (2030), 1.37 °C (2040)**; cubic 1.25 and 1.58 °C.

**`compare_degrees(t, y, degrees=(1, 2, 3)) -> DataFrame`**
In-sample comparison, index `degree`, columns `rmse`, `mae`, `r2`, `aic`, `bic`, `cond_X`, `lag1_autocorr` (lag-1 autocorrelation of the residuals; 0.83 for the line, 0.57 for the quadratic: the residuals are *not* independent).

**`split_by_year(t, y, cutoff) -> (t_train, y_train, t_test, y_test)`**
Train on years ≤ `cutoff`, test on the later years.

**`conditioning_table(t, y, degrees=(1, 2, 3)) -> DataFrame`**
Compares raw years vs centred time and normal equations vs QR. Columns: `degree`, `time`, `solver`, `cond(X)`, `cond(XtX)`, `fit_err` (largest difference in fitted values from the centred-QR reference, °C).
Real result: with raw years the quadratic has `cond(X) ≈ 9×10⁹`, `cond(XᵀX) ≈ 8×10¹⁹` (far beyond double precision); the cubic's normal-equation fit is off by up to 5×10⁻⁷ °C; with centred time `cond(X)` is 1.7–8 and errors are ~10⁻¹⁶.

**`forecast_spread(t, y, target_year=2040, degrees=(1, 2), starts=(1880, 1900, 1920, 1940, 1960, 1980)) -> DataFrame`**
The same model trained on different historical periods (first training year = `start`, training always ends at the last observation). Columns `degree`, `start`, `n_train`, `forecast` (for `target_year`), `slope_per_decade_at_end`. Windows with too few points are skipped.
Real result: linear 2040 forecast 0.81 °C (start 1880) → 1.36 °C (start 1980); quadratic 1.37 → 1.75 °C.

### 4.4 Plots

**`plot_fits(t, y, fits, future_end=2040, level=0.95, bands=True)`**: observations, each fitted curve extended to `future_end`, shaded prediction bands for the future, a dotted line at the last observation, and the final predicted value labelled on the right.
**`plot_forecast_spread(df, target_year=2040)`**: forecast for `target_year` against the first year of the training window, one line per degree.

### 4.5 Entry point

**`main()`**: fits linear and quadratic, saves `regression_fits.png`; saves the coefficient table (degrees 1–3), the 2030/2040/2050 forecasts, the in-sample comparison, the conditioning table, and the forecast-spread table and figure; prints the forecasts.

### 4.6 Tests (9)

| Test | What it checks |
|---|---|
| `test_design_matrix_shape_and_columns` | shape and that columns are 1, t, t² |
| `test_recovers_exact_polynomial` | noise-free quadratic data gives the exact coefficients |
| `test_solvers_agree_when_well_conditioned` | QR, `lstsq` and normal equations agree on centred data |
| `test_normal_equations_square_the_condition_number` | `cond(XᵀX) = cond(X)²` when centred; beyond 1/ε for raw years |
| `test_slope_and_extrapolation` | slope and 2030 prediction of a known line |
| `test_prediction_interval_widens_away_from_data` | interval width grows with distance |
| `test_interval_coverage_for_iid_noise` | with independent noise the 95 % interval covers 92–98 % |
| `test_conditioning_table_shows_normal_equations_degrade` | raw-year normal equations are ≥ 1000× worse |
| `test_forecast_spread_skips_windows_that_start_too_late` | short periods don't crash |

(`_line` is the test helper that builds a noisy straight-line series.)

### 4.7 Produces

`regression_fits.png`, `regression_coefficients.csv`, `regression_extrapolation.csv`, `regression_in_sample.csv`, `regression_conditioning.csv`, `regression_forecast_spread.csv` / `.png`.

---

## 5. Person 4: Validation + dashboard · `person4_validation.py`

**Role:** judge every method fairly, **only on data it has never seen**, and combine everything in one interactive app. Imports from all three other files.

### 5.1 Error scores

| Function | Formula | Notes |
|---|---|---|
| `_pair(y_true, y_pred)` | | *(private)* flattens to float arrays and checks the shapes match |
| `rmse(y_true, y_pred)` | `√mean(e²)` = `‖e‖₂/√n` | punishes large misses more |
| `mae(y_true, y_pred)` | `mean(|e|)` = `‖e‖₁/n` | average miss |
| `max_error(y_true, y_pred)` | `max|e|` = `‖e‖∞` | worst single miss |
| `bias(y_true, y_pred)` | `mean(pred − truth)` | negative = under-predicts |
| `summary(y_true, y_pred)` | | dict with `rmse`, `mae`, `max_error`, `bias` |

### 5.2 Test 1: interpolation on hidden years

**`holdout_experiment(years, values, methods=METHODS, frac=0.2, gap_length=1, n_trials=200, seed=0) -> DataFrame`**
Each trial hides a fraction of the series (`make_hidden_mask`), rebuilds it with every method (`predict_hidden`), and scores **only the hidden years**. One row per `(trial, method)`: `trial`, `method`, `gap_length`, `rmse`, `mae`.

**`summarize_holdout(results) -> DataFrame`**: mean and standard deviation of per-trial RMSE/MAE per method, best first.

**`interpolation_benchmark(years, values, gap_lengths=(1, 3, 5, 10), frac=0.2, n_trials=200, seed=0) -> DataFrame`**: runs the experiment for each gap length and returns mean `rmse`, `mae` per `(method, gap_length)`.
Real result (RMSE, °C): linear 0.087 (isolated) → 0.119 (10-year gaps); 4-point Lagrange 0.090 → 0.230; cubic spline 0.093 → 0.268.

**`noise_amplification(x, y, sigma=0.05, n_trials=50, seed=0, methods=None, global_nodes=25) -> DataFrame`**
Stability test. Adds `N(0, σ²)` noise to the known values and measures `max|change in curve| / max|noise|` on a fine grid, averaged over trials. 1 ≈ the noise passes through unchanged. Also tests global Lagrange on `global_nodes` equispaced nodes. Columns `nodes`, `amplification` (index = method).
Real result: linear 0.92, splines ≈ 1.0, **global Lagrange ≈ 13 000**.

### 5.3 Test 2: regression forecasting later years

**`cutoff_holdout(t, y, cutoff, degrees=(1, 2, 3), level=0.95) -> DataFrame`**: train up to `cutoff`, score on every later year. Columns `train_rmse`, `test_rmse`, `test_mae`, `test_bias`.
Real result (train 1880–2005, test 2006–2025): test RMSE linear 0.475, quadratic 0.187, cubic 0.150; the line's bias is −0.44 °C (it under-predicts because warming accelerated).

**`rolling_origin(t, y, degrees=(1, 2, 3), horizon=10, min_train=40, window=None, level=0.95) -> DataFrame`**
Forward-chaining backtest. For every origin year (starting with the 40th year) fit on the years before it (only the last `window` years if given) and forecast the next `horizon` years. One row per `(degree, origin, lead)`: `origin`, `lead`, `year`, `y_true`, `y_pred`, `lo`, `hi`. With the defaults: 970 forecasts per model.

**`summarize_backtest(bt, by="degree") -> DataFrame`**: `rmse`, `mae`, `bias`, `coverage` (fraction of true values inside the prediction interval) and `n` per group.
Real result: RMSE linear 0.236, **quadratic 0.168**, cubic 0.212; 95 % interval coverage only **68 % / 86 % / 80 %** (too narrow).

**`random_holdout(t, y, degrees=(1, 2, 3), frac=0.2, n_trials=200, seed=0)`**: hide random *interior* years, fit on the rest, score the hidden ones (the interpolation-style use of regression). Columns `trial`, `degree`, `rmse`, `mae`.

**`extrapolation_benchmark(t, y, degrees, horizon=10, min_train=40, window=None) -> (backtest, summary_by_degree_and_lead)`**: wraps `rolling_origin` and `summarize_backtest` by `["degree", "lead"]`.

**`training_period_sensitivity(t, y, degrees=(1, 2), starts=(1880, …, 1980), cutoffs=(1990, 2000, 2005, 2010, 2015), horizon=10) -> DataFrame`**
Fits on `[start, cutoff]` and scores the next `horizon` years. Columns `degree`, `start`, `cutoff`, `rmse`, `n_train`.
Real result: 10-year-ahead RMSE ranges from 0.06 to 0.84 °C depending on the window; the worst case is a quadratic fitted to only 11 years.

### 5.4 Test 3: SVD / matrix reconstruction on hidden entries

**`_time_interp_rows(A, mask, years, fn)`** *(private)*: rebuilds each region's hidden entries from **that region's own time series** using interpolator `fn`.

**`matrix_reconstruction_benchmark(A, years, ranks=(0, 1, 2, 3, 4), soft_lambdas=(1.0,), mask_kind="random", frac=0.2, gap_length=5, n_gaps=3, n_trials=30, seed=0) -> DataFrame`**
The head-to-head. Each trial hides entries of the matrix (`"random"` entries, or `"block"` runs of `gap_length` years, `n_gaps` per region) and reconstructs them by: per-region linear interpolation; per-region cubic spline; row mean (rank 0); SVD hard rank `k` for each `k` in `ranks`; SVD soft-threshold for each λ. Scores **only the hidden entries**. Columns `trial`, `method`, `rmse`, `mae`, `hidden_fraction`.

**`summarize(results, by="method") -> DataFrame`**: mean and std of RMSE/MAE per method.
Real result (RMSE, °C), random 20 % hidden / 5-year blocks: **linear in time 0.312 / 0.335**; SVD soft (λ=1) 0.358 / 0.345; SVD rank 1 0.414 / 0.394; row mean 0.585 / 0.563; SVD rank 3 0.783 / 0.836 (worse than the baseline: overfitting).

**`rank_generalisation(A, years, max_rank=6, frac=0.2, n_trials=30, seed=0) -> DataFrame`**
Index `k`; columns `in_sample_rmse` (error on the data kept: always falls as rank rises) and `held_out_rmse` (error on hidden entries: lowest at rank 1, then rises).

### 5.5 Plots

| Function | Shows |
|---|---|
| `plot_error_comparison(bench)` | grouped bars of RMSE and MAE per method for each gap length |
| `plot_backtest_by_lead(summary)` | left: forecast RMSE by years ahead; right: interval coverage against the nominal 95 % |
| `plot_training_sensitivity(df)` | heat maps (training start × forecast origin) of 10-year-ahead RMSE, one panel per degree |
| `_color(method)` *(private)* | blue for time-only methods, orange for SVD methods, gray for the row-mean baseline |
| `plot_method_comparison(summary, title)` | horizontal bars of held-out RMSE ± std, best at the top |
| `plot_noise_amplification(df)` | log-scale bars of noise amplification (global Lagrange highlighted) |

### 5.6 Dashboard

Run with `streamlit run person4_validation.py`. Everything it shows is computed by the real functions in the four files.

**Helpers**
- `show(fig)`: displays a Matplotlib figure in Streamlit and closes it.
- `cached_zonal()`, `cached_interp_benchmark(...)`, `cached_runge(...)`, `cached_backtest(...)`, `cached_matrix_benchmark(...)`, `cached_generalisation(...)`, `cached_sensitivity(...)`, `cached_noise(...)`: the slow computations wrapped in `@st.cache_data`, so dragging a slider is fast.

**`dashboard()`**: the app itself. A sidebar slider sets the **study period** (periods under 40 years are rejected with a message, because the hold-out tests need history). Five tabs:

| Tab | Controls | Shows |
|---|---|---|
| **Overview** | none | global anomaly graph, the last columns of matrix `A`, a note that the real data has no gaps |
| **1 · Interpolation** | fraction hidden, gap length, seed, method choice, number of trials | reconstruction graph + error table for that mask; benchmark bars; Runge/conditioning section |
| **2 · Regression** | degrees, extrapolate-to year, interval level, report years, training cutoff, backtest horizon / minimum training / window | fitted curves with bands; forecast table; equations, design-matrix rows, in-sample and conditioning tables; cutoff hold-out; rolling backtest |
| **3 · SVD** | centring mode, area weighting, rank `k`; gap type, hidden fraction/gap length, completion method (hard rank / soft threshold), `k` or λ, region | scree and error-vs-rank plots, original/reconstruction/residual heat maps, patterns, rank table with the Eckart–Young check; live matrix completion for any region |
| **4 · Validation** | mask type, hidden fraction, gap length, trials, forecast horizon, noise σ | method comparison, rank vs generalisation, training-period heat maps, 2040 forecast spread, noise amplification |

### 5.7 Entry point

**`main(args=None)`**: `python person4_validation.py` writes Person 4's tables and figures; with the argument **`all`** it first runs `person2_data.main()`, `person1_math.main()` and `person3_regression.main()`, so one command regenerates the whole project.

**`_in_streamlit()`** *(private)*: `True` when the file is run by `streamlit run`. At the bottom of the file: inside Streamlit it calls `dashboard()`, otherwise `main()`.

### 5.8 Tests (6)

| Test | What it checks |
|---|---|
| `test_metrics` | RMSE, MAE, max error and bias on a hand-computed example |
| `test_noise_amplification_global_lagrange_is_unstable` | linear ≤ 1.05, global Lagrange > 100 |
| `test_matrix_benchmark_ranks_models_sensibly_on_low_rank_data` | on rank-1 data, SVD rank 1 is > 3× better than the row mean |
| `test_training_sensitivity_and_spread_shapes` | the sensitivity and spread tables have the expected size |
| `test_holdout_experiment_runs_and_linear_exact_on_line` | linear interpolation is exact on a straight line |
| `test_rolling_origin_never_uses_future` | forecast years are always after the origin; leads are 1…horizon |

(`_line` is the test helper that builds a noisy straight-line series.)

### 5.9 Produces

`validation_interpolation_errors.csv` / `.png`, `validation_regression_holdout_after_2005.csv`, `validation_regression_random_holdout.csv`, `validation_regression_backtest.csv` / `.png`, `validation_svd_rank_generalisation.csv`, `validation_svd_error_vs_rank.png`, `validation_matrix_random.csv` / `.png`, `validation_matrix_block.csv` / `.png`, `validation_training_sensitivity.csv` / `.png`, `validation_noise_amplification.csv` / `.png`.

---

## 6. Output catalogue

Folders: `reports/figures/` (PNG) and `reports/tables/` (CSV), plus `data/processed/climate.csv`.

| Person | Figures | Tables |
|---|---|---|
| **2 · Data** | `data_trends`, `data_matrix` | `data/processed/climate.csv` |
| **1 · Math** | `interpolation_reconstruction`, `interpolation_runge`, `svd_scree`, `svd_modes`, `svd_reconstruction_rank2` | `interpolation_runge`, `svd_variance`, `svd_rank_table` |
| **3 · Regression** | `regression_fits`, `regression_forecast_spread` | `regression_coefficients`, `regression_extrapolation`, `regression_in_sample`, `regression_conditioning`, `regression_forecast_spread` |
| **4 · Validation** | `validation_interpolation_errors`, `validation_regression_backtest`, `validation_svd_error_vs_rank`, `validation_matrix_random`, `validation_matrix_block`, `validation_training_sensitivity`, `validation_noise_amplification` | `validation_interpolation_errors`, `validation_regression_holdout_after_2005`, `validation_regression_random_holdout`, `validation_regression_backtest`, `validation_svd_rank_generalisation`, `validation_matrix_random`, `validation_matrix_block`, `validation_training_sensitivity`, `validation_noise_amplification` |

16 figures and 17 tables in total.

---

## 7. Running, testing and extending

```bash
pip install -r requirements.txt
python person4_validation.py all     # everything (about 12 s)
python person2_data.py               # or one person at a time:  person1_math.py, person3_regression.py
streamlit run person4_validation.py  # dashboard
pytest                               # 39 tests (pytest.ini collects person*.py)
```

**Add an interpolation method (Person 1):** write `f(x_known, y_known, x_query) -> ndarray` and add it to `METHODS`. It automatically appears in the hidden-data tests, the plots and the dashboard (add a colour to `METHOD_COLOR`).

**Add a regression degree (Person 3):** `fit_polynomial(t, y, degree=4)` works as is; add `4` to the `degrees` tuples where you want it compared.

**Change the study period:** every `main()` and the dashboard take `start`/`end` (`load_global(start, end)`, or the sidebar slider).

**Use real gridded data later:** replace `region_matrix` in `person2_data.py` to return a larger `(regions, years)` matrix; the SVD, completion and benchmarks work on any shape.

---

## 8. Limitations

- The real table has no missing values, so every gap is artificial and randomly placed. Real gaps (for example sparse early polar records) may behave differently.
- Eight regions is a small matrix, which limits low-rank completion (ranks ≥ 3 overfit). Gridded GISTEMP data would be the natural next step.
- Extrapolation is curve-fitting, not a physical projection. The 95 % prediction intervals assume independent errors and are too narrow because the residuals are autocorrelated (lag-1 correlation 0.53–0.83); in backtests they covered only 68–86 % of outcomes.
- The 2040 forecast moves by about 0.4–0.55 °C with the choice of training window.
- The brief's "time plus a selected climate indicator" regression is not implemented: it needs a second dataset (for example CO₂ or an ENSO index). `design_matrix` could take extra columns, but none are included.
- Splines use natural end conditions only (no not-a-knot option).
- The second SVD pattern is dominated by the Antarctic band, where pre-1957 station coverage is sparse, so part of it may reflect data quality rather than climate.

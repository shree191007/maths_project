# ClimateMatrix: presentation script for four speakers

Every code snippet below is copied from this repository. Each is followed by what it does in plain words.
All numbers come from `reports/tables/` and can be regenerated with `python person4_validation.py all`.

**Speaking order (about 3 minutes each, 13 minutes in total):** Opening → Person 2 (data) → Person 1 (math) → Person 3 (regression) → Person 4 (validation + dashboard) → Closing.
The order follows how the data flows: *data → math → trends → testing*.

| | Person | Files | Output they bring | Pipeline command |
|---|---|---|---|---|
| 2 | Data | `person2_data.py` | `data/processed/climate.csv`, `reports/figures/data_trends.png`, `data_matrix.png` | `python person2_data.py` |
| 1 | Mathematics | `person1_math.py` | interpolation error table, scree plot, rank-vs-error table | `python person1_math.py` |
| 3 | Regression | `person3_regression.py` | `regression_fits.png`, `regression_extrapolation.csv` | `python person3_regression.py` |
| 4 | Validation + dashboard | `person4_validation.py` | `validation_*` tables and figures, the live dashboard | `python person4_validation.py` (add `all` to run everyone's), `streamlit run person4_validation.py` |

---

## Opening (any speaker, 30 seconds)

> "Our question is: **how well can we rebuild missing climate data, and how far can we trust a prediction of the future?**
> We use NASA's GISTEMP temperature record, 146 years across 8 latitude bands, and only linear algebra: interpolation to fill gaps, a
> matrix decomposition to find patterns, and least squares to fit trends.
> One distinction runs through the whole project. **Reconstruction** estimates values *inside* the period we observed.
> **Extrapolation** estimates values *outside* it. They are different tasks, and we will show they have very different accuracy."

---

## Person 2: Data (`person2_data.py`)

**Your sentence:** *"I collected and cleaned the climate dataset and prepared it for mathematical analysis."*

### What to say (about 3 minutes)

> "The data is NASA GISTEMP version 4, the zonal annual table: one temperature **anomaly** for each latitude band for every year from 1880 to 2025.
> An anomaly is the difference from the 1951–1980 average. Zero means 'normal for that baseline'; 1.2 means 1.2 °C warmer. Anomalies are used rather than raw temperatures because they are comparable across places that have very different absolute temperatures.
>
> I downloaded the file, loaded it with pandas and checked it before cleaning. It has 146 rows, 1880 to 2025, no duplicate years and no missing cells, but the NASA format marks a missing value as three stars, so my loader converts those to NaN, removes duplicate years and sorts by year. That means the loader still works if a future download contains gaps.
> Because the real data is complete, every 'missing' value in our project is one we **hide on purpose**. This lets us check each method against the truth.
>
> The file has overlapping summary columns, such as global, northern hemisphere and 24N–90N, plus 8 latitude bands that do not overlap. I used only the 8 bands for the matrix, because mixing in the summaries would make some rows exact combinations of others.
> The result is a matrix **A** with 8 rows (regions) and 146 columns (years). I saved the cleaned table as `data/processed/climate.csv`.
> Look at the heat map: the top row, the Arctic, turns dark red fastest. That is the pattern Person 1's SVD will find."

### Code walkthrough

**1. Loading and cleaning**
```python
def load_zonal(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        download_gistemp(path)
    df = pd.read_csv(path, na_values=["***", "****"])
    df = df.drop_duplicates("Year").set_index("Year").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")
```
- If the CSV isn't on disk it is downloaded from NASA first.
- `na_values=["***", "****"]` turns NASA's missing-value stars into real `NaN`s.
- `drop_duplicates("Year")` keeps one row per year, `set_index("Year").sort_index()` makes Year the row label in order.
- `to_numeric(errors="coerce")` forces every column to numbers; anything unparseable becomes `NaN` instead of crashing.

**2. Checking the raw file (the numbers you quote)**
```python
def quality_report(path: Path = DEFAULT_PATH) -> dict:
```
Returns `rows: 146, first_year: 1880, last_year: 2025, duplicate_years: 0, missing_cells: 0`, and anomaly range −2.59 to +3.24 °C. These are your "cleaning process" evidence.

**3. The eight non-overlapping regions**
```python
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
```
Each tuple is (name, south edge, north edge in degrees latitude). Together they tile the whole globe once. The edges also give each band's area weight, `(sin(north) − sin(south)) / 2`, via `band_area_weights()`.

**4. Building the region × year matrix**
```python
def region_matrix(df: pd.DataFrame, start: int | None = None, end: int | None = None, bands=BAND_NAMES):
    sub = df.loc[start:end, list(bands)]
    return sub.to_numpy(dtype=float).T, sub.index.to_numpy(), list(bands)
```
Selects the years you ask for and the 8 band columns, then **`.T` transposes** so rows are regions and columns are years. It returns `(A, years, names)`; `A.shape` is `(8, 146)`.

**5. Saving the cleaned data**
```python
    df[["Glob"] + BAND_NAMES].to_csv(dest)
```
`export_processed()` writes the global column and the 8 bands to `data/processed/climate.csv` (years as rows).

**6. The trend graph and the matrix picture**
`plot_trends(df)` draws the global anomaly as a thick black line over the eight bands, and `plot_matrix(A, years, names)` draws A as a red/blue heat map.

### Deliverable to show
`reports/figures/data_trends.png` (the trend graph) and `data_matrix.png`. The Arctic band moves most, from about −1.1 °C (1880s average) to about +2.9 °C recently, and the two polar bands swing the most overall (standard deviation 1.04 and 0.77 °C, against 0.29–0.64 elsewhere).

### Likely questions
- **Why anomalies, not temperatures?** They remove each place's baseline, so a band at the equator and one near a pole can be compared and combined.
- **Where did you handle missing values?** In `na_values` and `to_numeric(errors="coerce")`. This table has none, so we create gaps ourselves to test the methods.
- **Why 8 bands and not more regions?** The annual zonal table is what GISTEMP provides in one small CSV. Gridded data exists and would be the natural extension.

---

## Person 1: Mathematics (`person1_math.py`)

**Your sentence:** *"I implemented interpolation to estimate missing values and SVD to identify dominant patterns in climate data."*

### What to say (about 4 minutes)

> "I have two jobs. First, estimate a missing year from the years around it. Second, find the main patterns in the whole region-by-year matrix.
>
> **Interpolation** means finding a curve that passes exactly through the known points and reading it off at the missing year.
> *Linear* interpolation joins neighbouring points with a straight line. *Lagrange* interpolation builds a polynomial through several points; I use the 4 nearest, because a single polynomial through all 146 points blows up: with 40 nodes its error is millions of degrees.
> To test them I hide a fifth of the years, rebuild them, and compare with the truth. Surprisingly, **plain linear interpolation wins**: error 0.087 °C against 0.090 for Lagrange and 0.093 for a cubic spline, and the gap grows with longer missing stretches. Annual temperatures are a smooth trend plus about 0.1 °C of random noise, and the fancier curves chase that noise.
>
> **SVD** is the second part. We have the matrix A, 8 regions by 146 years. After subtracting each region's average, SVD factors it as **A = U Σ Vᵀ**. The columns of U are patterns across regions, the rows of Vᵀ are how those patterns change over time, and the singular values in Σ say how important each one is.
> The **rank** of a matrix is the number of independent patterns it contains; ours can have at most 8. But the first pattern alone explains 71.7% of the variance, and the first two explain 91.6%. So the data is *effectively* low rank: a few patterns describe everything.
> Pattern 1 is global warming, strongest in the Arctic. To *reconstruct* with fewer singular values we keep only the first k terms. Keeping 2 gives an average error of 0.17 °C; keeping 3 gives 0.11. This is the best possible rank-k approximation, and its error is exactly the size of the singular values we threw away."

### Code walkthrough

**1. Linear interpolation**
```python
    x, y, q = _check(x_known, y_known, x_query)
    i = np.clip(np.searchsorted(x, q, side="right"), 1, len(x) - 1)
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return y0 + (y1 - y0) * (q - x0) / (x1 - x0)
```
- `_check` validates the input, and refuses queries outside the known years, because interpolation must never extrapolate.
- `searchsorted` finds, for each missing year `q`, the first known year after it; `i-1` and `i` are the two known neighbours.
- The last line is the straight-line formula: start at `y0` and move `(q−x0)/(x1−x0)` of the way to `y1`.

**2. Lagrange interpolation**
```python
    out = np.zeros_like(q)
    for i in range(len(xn)):
        li = np.ones_like(q)
        for j in range(len(xn)):
            if j != i:
                li *= (q - xn[j]) / (xn[i] - xn[j])
        out += yn[i] * li
    return out
```
This is the formula **P(t) = Σ Tᵢ Lᵢ(t)** exactly. The inner loop builds the basis polynomial `Lᵢ(t) = Π (t − xⱼ)/(xᵢ − xⱼ)`, which equals 1 at known year `i` and 0 at every other known year. The outer loop adds up each known temperature times its basis polynomial.

**3. Local (windowed) Lagrange: why we don't use one big polynomial**
```python
        right = int(np.searchsorted(xs, qk))
        lo = int(np.clip(right - window // 2, 0, len(xs) - window))
        out[k] = _lagrange_eval(xs[lo:lo + window], y[lo:lo + window], np.array([qk]))[0]
```
For each missing year it picks the `window` nearest known years (we use 4 or 6) and interpolates only through those. `window=2` gives back linear interpolation.

**4. Hiding data and rebuilding it** (these two functions are yours; Person 4's `holdout_experiment` calls them in a loop)
```python
        hidden = make_hidden_mask(len(years), round(frac * len(years)), gap_length, rng)
        for name, pred in predict_hidden(years, values, hidden, methods).items():
```
`make_hidden_mask` marks random years (or runs of consecutive years) as hidden, never the first or last. `predict_hidden` rebuilds only those years from the rest, using every method. Person 4 repeats this 200 times and scores the results.

**5. SVD: centre, then decompose**
```python
    Ac, mu = center(A, center_mode)
    w = np.ones(A.shape[0]) if weights is None else np.asarray(weights, float)
    U, s, Vt = np.linalg.svd(Ac * w[:, None], full_matrices=False)
    return SVDResult(U, s, Vt, mu, w)
```
- `center` subtracts each region's mean (`mu = A.mean(axis=1, keepdims=True)`), so the SVD describes *variation*, not each region's offset.
- `weights` can optionally weight bands by area (default: all 1).
- `np.linalg.svd(...)` returns `U` (patterns over regions), `s` (singular values, largest first) and `Vt` (patterns over time).

**6. How much does each pattern explain?**
```python
    return s**2 / np.sum(s**2)
```
Squared singular value divided by the total: the share of the variance carried by each pattern. Result: 71.7%, 19.8%, 4.7%, 1.6%, 1.0%...

**7. Reconstruct with fewer singular values**
```python
    Ak = (res.U[:, :k] * res.s[:k]) @ res.Vt[:k]
    return Ak / res.weights[:, None] + res.mean
```
Keep only the first `k` columns of `U`, values of `s` and rows of `Vt`, multiply them back together, then add the region means back. This rebuilds a full 8×146 matrix from `k` patterns.

**8. Reconstruction error versus rank**
```python
        err = A - Ak
                     "frob_error": float(np.linalg.norm(err)),
                     "eckart_young": float(np.sqrt(np.sum(res.s[k:] ** 2))),
```
`frob_error` is the actual size of the leftover error. `eckart_young` is what the theory predicts: the square root of the sum of the squares of the singular values we discarded. **They match to machine precision at every rank** (a unit test checks it), which shows the truncation is the best possible rank-k approximation.

### Deliverable to show

| keep k patterns | variance explained | RMSE of rebuilt matrix | relative error |
|---|---|---|---|
| 1 | 71.7% | 0.313 °C | 53% |
| 2 | 91.6% | 0.171 °C | 29% |
| 3 | 96.3% | 0.114 °C | 19% |
| 8 (all) | 100% | 0 | 0% |

Plus `svd_scree.png` (explained variance), `svd_reconstruction_rank2.png` (original / rank-2 / residual) and `svd_modes.png`, and the interpolation error table:

| RMSE (°C) | isolated years | 10-year gaps |
|---|---|---|
| linear | **0.087** | **0.119** |
| Lagrange (4-point) | 0.090 | 0.230 |
| cubic spline | 0.093 | 0.268 |

### Likely questions
- **Why does the simplest method win?** The signal is smooth, but each year has about 0.1 °C of independent noise. A flexible curve fits that noise exactly and then carries it into the gap.
- **What is matrix rank?** The number of independent patterns in the matrix. Full rank here is 8; a rank-2 matrix is built from just two patterns.
- **Why subtract the mean first?** Without it the first pattern would just capture each region's baseline instead of how temperatures change.
- **Is the second pattern climate?** Mostly Antarctic variability, opposite to the Arctic. Antarctic station coverage before the late 1950s was sparse, so treat it with caution.

---

## Person 3: Regression (`person3_regression.py`)

**Your sentence:** *"I fitted linear and quadratic models to historical climate data and used them to estimate future temperature trends."*

### What to say (about 3 minutes)

> "My job is the trend and the future. **Regression** finds the curve through all the data, not necessarily through every point, that fits best overall. 'Best' means **least squares**: choose the coefficients that make the sum of squared errors as small as possible.
>
> I build a **design matrix X**. Each row is one year; the columns are 1, the year, and for the quadratic model the year squared. The target vector y holds the temperatures. Then the model is y ≈ Xβ and least squares finds β.
> The textbook answer is β = (XᵀX)⁻¹Xᵀy. We don't compute that. Squaring the matrix squares its condition number, and with raw years like 2020 the numbers are so lopsided that the normal equations lose all their precision. We rescale the years, then solve with a **QR decomposition**, which is stable.
>
> The results: a straight line fitted to 1880–2025 gives **0.72 °C for 2030 and 0.81 °C for 2040**. The quadratic gives **1.13 °C for 2030 and 1.37 °C for 2040**. They disagree because warming is speeding up, and a line can't see that. The line says 0.08 °C per decade; the quadratic says 0.22 °C per decade by today.
> I also compared different historical periods. Train the line on 1880–2025 and it predicts 0.81 °C for 2040; train it only on 1980–2025 and it predicts 1.36 °C. **The answer depends heavily on which years you use.**
> And the limitation: this is extrapolation. The curve is a shape, not physics. It assumes the recent pattern simply continues, and the uncertainty bands we draw are too narrow, which Person 4 will quantify."

### Code walkthrough

**1. The design matrix X**
```python
    tau = (np.asarray(t, dtype=float) - t_center) / t_scale
    return np.vander(tau, degree + 1, increasing=True)
```
First the years `t` are shifted and scaled to the range −1…1 (`tau`). `np.vander(..., increasing=True)` then builds the columns `[1, τ, τ², …]`. Degree 1 gives 2 columns (linear), degree 2 gives 3 (quadratic).

**2. Solving least squares (QR, not the inverse)**
```python
    t_center, t_scale = ((t.min() + t.max()) / 2, (t.max() - t.min()) / 2) if center else (0.0, 1.0)
    X = design_matrix(t, degree, t_center, t_scale)
    Q, R = qr(X, mode="economic")
    if solver == "qr":
        coef = solve_triangular(R, Q.T @ y)
```
- The first line chooses the centre and scale from the data, so years become −1…1.
- `qr` factors `X = QR` with `Q` orthogonal and `R` triangular. Least squares then reduces to the easy triangular system `Rβ = Qᵀy`, solved by `solve_triangular`. That is the stable way to get the same answer as `(XᵀX)⁻¹Xᵀy`.

**3. The textbook formula, kept only for comparison**
```python
        coef = np.linalg.solve(X.T @ X, X.T @ y)  # textbook formula, squares cond(X)
```
This is the normal-equations option. `conditioning_table()` compares the two: with raw years the quadratic's `cond(XᵀX)` is about 8×10¹⁹, far past what double precision can resolve.

**4. Prediction with an uncertainty band**
```python
        mean = X @ self.coef
        se = np.sqrt(self.sigma2 * (1.0 + np.einsum("ij,jk,ik->i", X, self.cov_unscaled, X)))
        half = stats.t.ppf(0.5 + level / 2, dof) * se
        return mean, mean - half, mean + half
```
`X @ coef` is the prediction. `se` is the standard error of a new observation (it grows further from the data), and `half` multiplies it by the t-distribution cut-off for the chosen level (95% by default). It returns the prediction and lower/upper bounds.

**5. Extrapolating to future years**
```python
    mean, lo, hi = fit.predict(years, level)
    return pd.DataFrame({"year": years.astype(int), "prediction": mean, "lo": lo, "hi": hi})
```
`extrapolate(fit, [2030, 2040])` gives a small table of prediction with lower/upper bounds for each future year.

**6. Comparing different historical periods**
```python
# usage example (not in a file): same model, different training window
for start in (1880, 1940, 1980):
    keep = years >= start
    fit = fit_polynomial(years[keep], values[keep], degree=1)
    print(start, fit.predict([2040]))
```
`forecast_spread()` (also in `person3_regression.py`) does exactly this for linear and quadratic over six start years; results are in `reports/tables/regression_forecast_spread.csv`.

**7. Testing on later years**
```python
    tr_t, tr_y, te_t, te_y = split_by_year(t, y, cutoff)
    f = fit_polynomial(tr_t, tr_y, d)
```
`cutoff_holdout(t, y, 2005)` (in Person 4's file, built on Person 3's `split_by_year` and `fit_polynomial`) trains only on years up to 2005 and scores the model on 2006–2025, years it never saw.

### Deliverable to show
`regression_fits.png` (history, both fitted curves, 95% bands extended to 2040) and:

| model | 2030 | 2040 | RMSE on history |
|---|---|---|---|
| linear | 0.72 °C (0.33–1.11) | 0.81 °C (0.42–1.20) | 0.193 |
| quadratic | 1.13 °C (0.88–1.38) | 1.37 °C (1.11–1.62) | 0.120 |

2040 forecast by where training starts: linear 0.81 (1880) → 0.98 (1920) → 1.07 (1940) → 1.36 (1980); quadratic 1.37 (1880) → 1.75 (1980).

### Likely questions
- **What's the difference between your fit and Person 1's interpolation?** Interpolation passes *through* every known point and is used inside the data. Regression smooths *across* them, and we use it to look beyond the data.
- **Why is the quadratic better than the linear?** Warming accelerated. Trained on 1880–2005, the line under-predicted 2006–2025 by 0.44 °C on average, the quadratic by 0.14 °C.
- **So is the 2040 forecast reliable?** No. It is a curve extended, not a climate projection. It changes by 0.4–0.55 °C just from the training window we choose.

---

## Person 4: Validation + dashboard (`person4_validation.py`)

**Your sentence:** *"I evaluated the methods using error metrics and integrated the results into an interactive dashboard."*

### What to say (about 3 minutes)

> "I test everything, and the rule is simple: **only score a method on data it has not seen.**
> I use two metrics. **MAE** is the average size of the mistake. **RMSE** squares the mistakes first, so it punishes big misses more. Both are in °C.
>
> *Interpolation:* hide years, rebuild, score. Linear interpolation is best, with an RMSE of 0.09 °C for single missing years and 0.12 °C for 10-year gaps.
> *Regression on later years:* I run a **rolling forecast**. Pretend it is 1919, fit the model, forecast the next 10 years, score it, then move forward one year and repeat, 970 forecasts per model. The quadratic is the best: RMSE 0.17 °C against 0.24 for the line. The cubic fits history slightly better but forecasts *worse*, so a better fit does not mean a better prediction.
> *SVD:* I hide matrix entries and rebuild them from the other regions. Error on the data we kept always falls as rank rises, but error on the hidden entries is smallest at rank 1 and gets worse from rank 3. Too many patterns means fitting noise.
>
> So, reconstruction versus extrapolation: rebuilding missing years is accurate to about 0.09–0.12 °C. Forecasting 1–10 years ahead is about 0.17 °C at best, and our 95% bands only contained the truth **68–86% of the time**, so they are overconfident. And the 2040 estimate shifts with the training period.
> Everything is in one dashboard. Let me show it."

### Code walkthrough

**1. The two error metrics (section 1 of `person4_validation.py`)**
```python
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(np.mean(np.abs(y_true - y_pred)))
```
First line is **RMSE**: square each error, average, take the square root. Second is **MAE**: average of the absolute errors. The same section also has `max_error` (the worst miss) and `bias` (does the model tend to over- or under-predict).

**2. Testing interpolation on hidden data**
```python
        hidden = make_hidden_mask(len(years), round(frac * len(years)), gap_length, rng)
        for name, pred in predict_hidden(years, values, hidden, methods).items():
            rows.append({"trial": trial, "method": name, "gap_length": gap_length,
                         "rmse": rmse(values[hidden], pred), "mae": mae(values[hidden], pred)})
```
Each trial hides a random fraction, lets every interpolation method rebuild it, and records RMSE and MAE **only on the hidden years**. `interpolation_benchmark` repeats this for gap lengths 1, 3, 5 and 10.

**3. Testing regression on later years (rolling forecast)**
```python
    for end in range(min_train, len(t) - horizon + 1):
            f = fit_polynomial(t[lo_i:end], y[lo_i:end], d)
            ft, fy = t[end:end + horizon], y[end:end + horizon]
```
`end` moves forward one year at a time. Each time the model is fitted on years **before** `end` and asked to predict the next `horizon` (10) years, which it has not seen. `summarize_backtest` then reports RMSE, MAE, bias and how often the 95% band contained the truth.

**4. Testing the SVD reconstruction**
```python
        mask = random_mask(A.shape, frac, rng) if mask_kind == "random" else block_mask(A.shape, gap_length, n_gaps, rng)
        A_obs = np.where(mask, np.nan, A)
        for k in ranks:
            preds[label] = impute_lowrank(A_obs, k)[mask]
```
Hide entries of the matrix (random ones, or runs of consecutive years), set them to `NaN`, fill them back in using a rank-`k` SVD for each `k`, and score on the hidden entries only. The same masks are also given to per-region linear and spline interpolation, so every method is compared fairly.

**5. The dashboard (`dashboard()` in `person4_validation.py`)**
```python
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Overview", "1 · Interpolation", "2 · Regression", "3 · SVD", "4 · Validation"])
```
Five tabs: the data overview, then one tab per part, each with sliders (study period, hidden fraction, gap length, model degree, rank, and so on) that re-run the real code.
```python
@st.cache_data
```
Expensive benchmarks are cached so moving a slider is fast. A study period under 40 years is rejected with a message, because the hold-out tests need enough history.

**Live demo order (2 minutes):** 1 · Interpolation tab (hide 20%, show the error table) → 2 · Regression tab (tick quadratic, drag "Extrapolate to" to 2040) → 3 · SVD tab (drag Rank k from 0 to 8, watch the reconstruction sharpen) → 4 · Validation tab (method comparison and training-period heat map).

### Deliverable to show

| Test | Best method | Error |
|---|---|---|
| Interpolation, isolated years | linear | RMSE 0.087 °C |
| Interpolation, 10-year gaps | linear | RMSE 0.119 °C |
| Forecast next 10 years (rolling) | quadratic | RMSE 0.168 °C |
| Matrix gaps, random 20% | linear in time (0.312); SVD soft-threshold 0.358; SVD rank 1 0.414 | |
| Matrix gaps, 5-year blocks | linear in time (0.335); SVD soft-threshold 0.345; SVD rank 1 0.394 | |

SVD rank vs error: in-sample RMSE 0.31 → 0.17 → 0.11 for k = 1, 2, 3, but on hidden entries it is 0.41, 0.52, 0.78. Row-mean baseline 0.58.

### Limitations to state
- The real table has no gaps, so every gap is artificial and randomly placed. Real gaps (sparse early polar data) may behave differently.
- 8 regions is a small matrix, which limits what low-rank completion can do.
- Extrapolation is curve-fitting, not a physical projection; the 95% bands assume independent errors and are too narrow because yearly residuals are strongly autocorrelated (lag-1 correlation 0.53–0.83).
- Not included: regression on a second climate indicator (such as CO₂), which would need another dataset.

### Likely questions
- **Why both RMSE and MAE?** If they are far apart, a few large misses dominate. They give the same ranking here.
- **Why not test on the data you fitted?** That only measures how well a method memorises. The cubic fits history best and forecasts worse.
- **What would you do next?** Gridded GISTEMP data (thousands of rows), a second predictor for regression, and bands that account for autocorrelation.

---

## Closing (30 seconds, any speaker)

> "To answer our question. **Reconstruction works well**: missing years are recovered to about 0.1 °C, and two patterns explain 92% of how regional temperatures vary.
> **Extrapolation is much less reliable**: the best forecast is about 0.17 °C off, the uncertainty bands are overconfident, and the 2040 estimate moves with the training window.
> Reconstruction and extrapolation are different tasks, and a method that is good at one does not guarantee it is good at the other."

---

## Checklist: what each person must have ready

| | Working code | One graph or table | Explanation of the method | Short talk |
|---|---|---|---|---|
| Person 2 | `person2_data.py` | `data_trends.png` or `data_matrix.png` | cleaning steps, what an anomaly is | 3 min |
| Person 1 | `person1_math.py` | error table + `svd_scree.png` | interpolation, SVD, rank, reconstruction error | 4 min |
| Person 3 | `person3_regression.py` | `regression_fits.png` | least squares, QR vs inverse, extrapolation limits | 3 min |
| Person 4 | `person4_validation.py` | comparison tables + the dashboard | RMSE/MAE, hidden-data testing, limitations | 3 min |

Before presenting, run once:
```bash
python person4_validation.py all
streamlit run person4_validation.py
```

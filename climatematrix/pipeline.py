"""Run each part end-to-end and write its deliverables to reports/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data, interpolation as ip, regression as rg, svd_analysis as sv, validation as va
from . import plot_interpolation as p1, plot_regression as p2, plot_svd as p3, plot_validation as p4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def to_md(df: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    d = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    cols = [str(c) for c in d.columns]
    rows = [[fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def _save(fig, name: str, out: Path):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"{name}.png", dpi=150)
    plt.close(fig)


def _table(df, name: str, out: Path, **kw):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / f"{name}.csv", **kw)


def load(start: int = 1880, end: int | None = None):
    df = data.load_zonal()
    g = data.global_series(df, start, end)
    return df, g.index.to_numpy(float), g.to_numpy()


def run_part1(out: Path = OUT, start=1880, end=None, n_trials=200) -> dict:
    df, yrs, v = load(start, end)
    res = {}
    # Deliverable 1: graph of actual vs reconstructed values
    rng = np.random.default_rng(7)
    hidden = ip.make_hidden_mask(len(yrs), round(0.2 * len(yrs)), 1, rng)
    preds = ip.predict_hidden(yrs, v, hidden)
    _save(p1.plot_reconstruction(yrs, v, hidden, preds), "p1_reconstruction", out)
    # Deliverable 2: table of interpolation errors (random isolated gaps and block gaps)
    bench = va.interpolation_benchmark(yrs, v, n_trials=n_trials)
    table = bench.pivot(index="method", columns="gap_length", values="rmse")
    table.columns = [f"RMSE, gap={c}" for c in table.columns]
    mae = bench.pivot(index="method", columns="gap_length", values="mae")
    mae.columns = [f"MAE, gap={c}" for c in mae.columns]
    res["errors"] = table.join(mae).sort_values(table.columns[0])
    _table(res["errors"], "p1_interpolation_errors", out)
    _save(p1.plot_error_comparison(bench), "p1_error_comparison", out)
    res["runge"] = ip.runge_experiment(yrs, v)
    _table(res["runge"], "p1_runge", out, index=False)
    _save(p1.plot_runge(res["runge"]), "p1_runge", out)
    return res


def run_part2(out: Path = OUT, start=1880, end=None, cutoff=2005) -> dict:
    df, yrs, v = load(start, end)
    res = {}
    fits = [rg.fit_polynomial(yrs, v, d) for d in (1, 2)]
    _save(p2.plot_fits(yrs, v, fits, 2040), "p2_fits", out)
    rows = []
    for f in [rg.fit_polynomial(yrs, v, d) for d in (1, 2, 3)]:
        for k, c in enumerate(f.coef):
            rows.append({"degree": f.degree, "coef": f"c{k}", "value": c})
    res["coefficients"] = pd.DataFrame(rows).pivot(index="degree", columns="coef", values="value")
    _table(res["coefficients"], "p2_coefficients", out)
    ex = []
    for f in fits:
        e = rg.extrapolate(f, [2030, 2040, 2050])
        e.insert(0, "model", {1: "linear", 2: "quadratic"}[f.degree])
        ex.append(e)
    res["extrapolation"] = pd.concat(ex, ignore_index=True)
    _table(res["extrapolation"], "p2_extrapolation", out, index=False)
    res["in_sample"] = rg.compare_degrees(yrs, v)
    _table(res["in_sample"], "p2_in_sample", out)
    res["cutoff"] = rg.cutoff_holdout(yrs, v, cutoff)
    _table(res["cutoff"], f"p2_holdout_after_{cutoff}", out)
    res["random_holdout"] = rg.random_holdout(yrs, v).groupby("degree")[["rmse", "mae"]].mean()
    _table(res["random_holdout"], "p2_random_holdout", out)
    bt, summ = va.extrapolation_benchmark(yrs, v)
    res["backtest"] = rg.summarize_backtest(bt, "degree")
    _table(res["backtest"], "p2_backtest", out)
    _save(p2.plot_backtest_by_lead(summ), "p2_backtest", out)
    res["conditioning"] = rg.conditioning_table(yrs, v)
    _table(res["conditioning"], "p2_conditioning", out, index=False)
    return res


def run_part3(out: Path = OUT, start=1880, end=None, weighted: bool = False) -> dict:
    df = data.load_zonal()
    A, yrs, names = data.region_matrix(df, start, end)
    w = np.sqrt(data.band_area_weights()) if weighted else None
    r = sv.compute_svd(A, "row", w)
    res = {"svd": r, "A": A, "years": yrs, "names": names}
    res["rank_table"] = sv.rank_table(A, r)
    _table(res["rank_table"], "p3_rank_table", out)
    res["variance"] = pd.DataFrame({"explained": sv.explained_variance(r.s), "cumulative": np.cumsum(sv.explained_variance(r.s)),
                                    "singular_value": r.s}, index=pd.RangeIndex(1, len(r.s) + 1, name="component"))
    _table(res["variance"], "p3_variance", out)
    _save(p3.plot_scree(r.s), "p3_scree", out)
    _save(p3.plot_modes(r, names, yrs), "p3_modes", out)
    _save(p3.plot_original_vs_reconstructed(A, sv.reconstruct(r, 2), yrs, names, 2), "p3_reconstruction_rank2", out)
    res["generalisation"] = va.rank_generalisation(A, yrs)
    _table(res["generalisation"], "p3_rank_generalisation", out)
    _save(p3.plot_error_vs_rank(res["rank_table"].loc[:6], res["generalisation"]), "p3_error_vs_rank", out)
    return res


def run_part4(out: Path = OUT, start=1880, end=None, n_trials=30) -> dict:
    df, yrs, v = load(start, end)
    A, ayrs, names = data.region_matrix(df, start, end)
    res = {}
    for kind, kw in (("random", {"frac": 0.2}), ("block", {"gap_length": 5, "n_gaps": 3})):
        bench = va.matrix_reconstruction_benchmark(A, ayrs, mask_kind=kind, n_trials=n_trials, **kw)
        res[f"matrix_{kind}"] = va.summarize(bench).sort_values("rmse")
        _table(res[f"matrix_{kind}"], f"p4_matrix_{kind}", out)
        _save(p4.plot_method_comparison(res[f"matrix_{kind}"], f"Region×year matrix, {kind} gaps: held-out RMSE"),
              f"p4_matrix_{kind}", out)
    res["sens"] = va.training_period_sensitivity(yrs, v)
    _table(res["sens"], "p4_training_sensitivity", out, index=False)
    _save(p2.plot_training_sensitivity(res["sens"]), "p4_training_sensitivity", out)
    res["spread"] = va.forecast_spread(yrs, v)
    _table(res["spread"], "p4_forecast_spread", out, index=False)
    _save(p2.plot_forecast_spread(res["spread"]), "p4_forecast_spread", out)
    res["noise"] = va.noise_amplification(yrs, v)
    _table(res["noise"], "p4_noise_amplification", out)
    _save(p4.plot_noise_amplification(res["noise"]), "p4_noise_amplification", out)
    return res


def run_all(out: Path = OUT) -> dict:
    return {"part1": run_part1(out), "part2": run_part2(out), "part3": run_part3(out), "part4": run_part4(out)}

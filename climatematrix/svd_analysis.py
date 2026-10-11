"""Part 3: SVD of the region x year anomaly matrix.

A (regions x years) is centred, then A_c = U S V^T. Column i of U is a spatial
pattern over regions, row i of V^T is a time series, s_i says how much
variance that pair carries. The rank-k truncation A_k = U_k S_k V_k^T is the
best rank-k approximation in the Frobenius norm (Eckart-Young), with error
sqrt(sum_{i>k} s_i^2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CENTER_MODES = ("row", "col", "none")


@dataclass
class SVDResult:
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    mean: np.ndarray        # what was subtracted (broadcastable to A)
    weights: np.ndarray     # row weights applied before the SVD (ones if unweighted)

    @property
    def rank(self) -> int:
        return len(self.s)


def center(A: np.ndarray, mode: str = "row"):
    """Subtract row means ('row': each region's own time-mean), column means
    ('col': the cross-region mean of each year) or nothing."""
    if mode not in CENTER_MODES:
        raise ValueError(f"mode must be one of {CENTER_MODES}")
    if mode == "row":
        mu = A.mean(axis=1, keepdims=True)
    elif mode == "col":
        mu = A.mean(axis=0, keepdims=True)
    else:
        mu = np.zeros((1, 1))
    return A - mu, mu


def compute_svd(A: np.ndarray, center_mode: str = "row", weights: np.ndarray | None = None) -> SVDResult:
    """Thin SVD of the centred (and optionally row-weighted) matrix.

    ``weights`` are per-row multipliers, e.g. sqrt(area fraction) so polar
    bands, which cover little area, do not dominate.
    """
    if np.isnan(A).any():
        raise ValueError("A contains NaN: use impute_lowrank for incomplete matrices")
    Ac, mu = center(A, center_mode)
    w = np.ones(A.shape[0]) if weights is None else np.asarray(weights, float)
    U, s, Vt = np.linalg.svd(Ac * w[:, None], full_matrices=False)
    return SVDResult(U, s, Vt, mu, w)


def explained_variance(s: np.ndarray) -> np.ndarray:
    """Fraction of total squared Frobenius norm carried by each component."""
    return s**2 / np.sum(s**2)


def reconstruct(res: SVDResult, k: int) -> np.ndarray:
    """Rank-k approximation of the original (un-centred, un-weighted) matrix."""
    Ak = (res.U[:, :k] * res.s[:k]) @ res.Vt[:k]
    return Ak / res.weights[:, None] + res.mean


def spatial_pattern(res: SVDResult, i: int) -> np.ndarray:
    """Mode i over regions, in physical units (undoing the row weights),
    scaled so its largest magnitude is 1 and its sign is fixed (sum >= 0)."""
    u = res.U[:, i] / res.weights
    u = u / np.max(np.abs(u))
    return u if u.sum() >= 0 else -u


def temporal_amplitude(res: SVDResult, i: int) -> np.ndarray:
    """Time series paired with ``spatial_pattern(res, i)`` (same sign convention)."""
    u = res.U[:, i] / res.weights
    sign = 1.0 if (u / np.max(np.abs(u))).sum() >= 0 else -1.0
    return sign * res.s[i] * res.Vt[i] * np.max(np.abs(u))


def rank_table(A: np.ndarray, res: SVDResult, ks=None) -> pd.DataFrame:
    """Reconstruction error versus retained rank, with the Eckart-Young check.

    ``frob_error`` is measured on the original matrix; ``eckart_young`` is the
    prediction sqrt(sum_{i>k} s_i^2) which equals it exactly for an unweighted fit.
    """
    ks = range(0, res.rank + 1) if ks is None else ks
    ev = explained_variance(res.s)
    total = np.linalg.norm(A - res.mean)
    rows = []
    for k in ks:
        Ak = reconstruct(res, k) if k > 0 else np.broadcast_to(res.mean, A.shape).copy()
        err = A - Ak
        rows.append({"k": k, "explained_cum": float(ev[:k].sum()),
                     "frob_error": float(np.linalg.norm(err)),
                     "eckart_young": float(np.sqrt(np.sum(res.s[k:] ** 2))),
                     "rel_error": float(np.linalg.norm(err) / total),
                     "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows).set_index("k")


def numerical_rank(s: np.ndarray, tol: float | None = None) -> int:
    """Number of singular values above tol (default: eps * max(shape) * s_max)."""
    tol = np.finfo(float).eps * s.max() * len(s) if tol is None else tol
    return int(np.sum(s > tol))


# --------------------------------------------------------- matrix completion
def random_mask(shape, frac: float, rng: np.random.Generator) -> np.ndarray:
    """True where entries are hidden; first/last column kept for fair comparison
    with interpolation (which cannot extrapolate)."""
    m = rng.random(shape) < frac
    m[:, [0, -1]] = False
    return m


def block_mask(shape, gap_length: int, n_gaps_per_region: int, rng: np.random.Generator) -> np.ndarray:
    """Each region loses ``n_gaps_per_region`` runs of ``gap_length`` consecutive years."""
    m = np.zeros(shape, dtype=bool)
    for r in range(shape[0]):
        placed, attempts = 0, 0
        while placed < n_gaps_per_region and attempts < 1000:
            attempts += 1
            start = int(rng.integers(1, shape[1] - gap_length))
            lo, hi = max(start - 1, 0), min(start + gap_length + 1, shape[1])
            if not m[r, lo:hi].any():
                m[r, start:start + gap_length] = True
                placed += 1
    return m


def _complete(A_obs: np.ndarray, truncate, max_iter: int, tol: float):
    """Shared loop: fill gaps, re-centre by row, low-rank-ify with ``truncate``,
    write the result into the gaps only, repeat until the update is below tol."""
    obs = ~np.isnan(A_obs)
    if not obs.any(axis=1).all():
        raise ValueError("every region needs at least one observation")
    X = np.where(obs, A_obs, np.nanmean(A_obs, axis=1, keepdims=True))
    history = []
    for _ in range(max_iter):
        mu = X.mean(axis=1, keepdims=True)
        X_new = np.where(obs, A_obs, truncate(X - mu) + mu)
        history.append(np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-12))
        X = X_new
        if history[-1] < tol:
            break
    return X, history


def impute_lowrank(A_obs: np.ndarray, k: int, max_iter: int = 500, tol: float = 1e-8, return_history: bool = False):
    """Fill NaNs with a rank-k + row-mean model (iterative hard-impute).

    Each pass rebuilds the matrix from its rank-k SVD and replaces *only the
    missing entries*. k=0 is the plain row-mean fill (baseline). With only a
    handful of regions, k >= 3 overfits badly: a year with several gaps leaves
    fewer observed values than free parameters.
    """
    def truncate(Xc):
        if k == 0:
            return np.zeros_like(Xc)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :k] * s[:k]) @ Vt[:k]

    X, history = _complete(A_obs, truncate, max_iter, tol)
    return (X, history) if return_history else X


def impute_soft(A_obs: np.ndarray, lam: float, max_iter: int = 500, tol: float = 1e-8, return_history: bool = False):
    """Soft-threshold completion: shrink every singular value s -> max(s - lam, 0).

    A smooth alternative to choosing a hard rank. ``lam`` is in the units of
    the singular values (deg C times sqrt of matrix size); lam=0 keeps all
    components and overfits just like full rank.
    """
    def truncate(Xc):
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U * np.maximum(s - lam, 0.0)) @ Vt

    X, history = _complete(A_obs, truncate, max_iter, tol)
    return (X, history) if return_history else X

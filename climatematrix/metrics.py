"""Error norms shared by every module (owned by Part 4)."""
from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error, ||e||_2 / sqrt(n)."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, ||e||_1 / n."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def max_error(y_true, y_pred) -> float:
    """Worst-case error, ||e||_inf."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def bias(y_true, y_pred) -> float:
    """Mean signed error (prediction minus truth)."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def summary(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }

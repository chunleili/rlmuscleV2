"""Per-muscle comparison metrics between two activation series.

Pure functions on numpy arrays; no GT-source knowledge. Used by the harness
to score learned activations against ground truth."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def pearson_r(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / denom) if denom > _EPS else 0.0


def rmse(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0


def timing_lag(x, y, dt) -> float:
    """Seconds by which y lags x (argmax of normalized cross-correlation)."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx < _EPS or ny < _EPS:
        return 0.0
    corr = np.correlate(y / ny, x / nx, mode="full")
    lag_samples = np.argmax(corr) - (len(x) - 1)
    return float(lag_samples * dt)


def co_contraction_index(a, b) -> float:
    """Falconer-Winter co-contraction: mean_t 2*min(a,b)/(a+b). Inputs >= 0."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = a + b
    cci = np.where(s > _EPS, 2.0 * np.minimum(a, b) / np.maximum(s, _EPS), 0.0)
    return float(np.mean(cci))

"""Regression and correlation metrics as pure functions."""

from collections.abc import Sequence

import numpy as np


def _paired(
    predicted: Sequence[float], reference: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(predicted, dtype=float)
    y_true = np.asarray(reference, dtype=float)
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("predicted and reference must be 1-dimensional")
    if len(y_pred) != len(y_true):
        raise ValueError(f"length mismatch: {len(y_pred)} vs {len(y_true)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on zero examples")
    return y_pred, y_true


def mse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    return float(np.sqrt(mse(predicted, reference)))


def mae(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Mean absolute percentage error. Undefined when any reference is zero."""
    y_pred, y_true = _paired(predicted, reference)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when a reference value is zero")
    return float(np.mean(np.abs((y_pred - y_true) / y_true)))


def smape(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Symmetric MAPE in [0, 2], using the |pred|+|true| denominator variant."""
    y_pred, y_true = _paired(predicted, reference)
    denom = np.abs(y_pred) + np.abs(y_true)
    if np.any(denom == 0):
        raise ValueError("sMAPE is undefined when pred and true are both zero")
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def r2_score(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("R^2 is undefined for a constant reference")
    return 1.0 - ss_res / ss_tot


def explained_variance(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    var_true = float(np.var(y_true))
    if var_true == 0.0:
        raise ValueError("explained variance is undefined for a constant reference")
    return 1.0 - float(np.var(y_true - y_pred)) / var_true


def pearson_r(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        raise ValueError("Pearson r is undefined for a constant input")
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), with ties sharing the mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.mean(np.arange(i + 1, j + 2)))
        i = j + 1
    return ranks


def spearman_rho(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = _paired(predicted, reference)
    r_pred, r_true = _rank(y_pred), _rank(y_true)
    if np.std(r_pred) == 0 or np.std(r_true) == 0:
        raise ValueError("Spearman rho is undefined for a constant input")
    return float(np.corrcoef(r_pred, r_true)[0, 1])


def kendall_tau(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Kendall's tau-b (accounts for ties), O(n^2) reference implementation."""
    y_pred, y_true = _paired(predicted, reference)
    n = len(y_pred)
    concordant = discordant = ties_pred = ties_true = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = np.sign(y_pred[i] - y_pred[j])
            dt = np.sign(y_true[i] - y_true[j])
            if dp == 0 and dt == 0:
                continue
            if dp == 0:
                ties_pred += 1
            elif dt == 0:
                ties_true += 1
            elif dp == dt:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt(
        (concordant + discordant + ties_pred) * (concordant + discordant + ties_true)
    )
    if denom == 0:
        raise ValueError("Kendall tau is undefined for a constant input")
    return float((concordant - discordant) / denom)

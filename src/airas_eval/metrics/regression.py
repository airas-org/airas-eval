"""Regression and correlation metrics, delegated to scikit-learn / scipy.

In-house exceptions, kept deliberately:
- ``mape`` raises on zero references instead of scikit-learn's silent epsilon
  clamping (which quietly turns the metric into a different one);
- ``smape`` has no sklearn implementation; the |pred|+|true| denominator
  variant is fixed here.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats as _stats
from sklearn import metrics as _skm

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics._validate import paired_1d


def mse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    return float(_skm.mean_squared_error(y_true, y_pred))


def rmse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    return float(_skm.root_mean_squared_error(y_true, y_pred))


def mae(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    return float(_skm.mean_absolute_error(y_true, y_pred))


def mape(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Strict MAPE: undefined (raises) when any reference value is zero."""
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if np.any(y_true == 0):
        raise UndefinedMetric("MAPE is undefined when a reference value is zero")
    return float(np.mean(np.abs((y_pred - y_true) / y_true)))


def smape(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Symmetric MAPE in [0, 2], using the |pred|+|true| denominator variant."""
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    denom = np.abs(y_pred) + np.abs(y_true)
    if np.any(denom == 0):
        raise UndefinedMetric("sMAPE is undefined when pred and true are both zero")
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def r2_score(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if float(np.var(y_true)) == 0.0:
        raise UndefinedMetric("R^2 is undefined for a constant reference")
    return float(_skm.r2_score(y_true, y_pred))


def explained_variance(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if float(np.var(y_true)) == 0.0:
        raise UndefinedMetric(
            "explained variance is undefined for a constant reference"
        )
    return float(_skm.explained_variance_score(y_true, y_pred))


def pearson_r(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        raise UndefinedMetric("Pearson r is undefined for a constant input")
    return float(_stats.pearsonr(y_pred, y_true).statistic)


def spearman_rho(predicted: Sequence[float], reference: Sequence[float]) -> float:
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        raise UndefinedMetric("Spearman rho is undefined for a constant input")
    return float(_stats.spearmanr(y_pred, y_true).statistic)


def kendall_tau(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Kendall's tau-b (scipy default variant)."""
    y_pred, y_true = paired_1d(predicted, reference, dtype=float)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        raise UndefinedMetric("Kendall tau is undefined for a constant input")
    return float(_stats.kendalltau(y_pred, y_true).statistic)

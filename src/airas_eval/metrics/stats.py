"""Statistical reporting helpers.

Papers must report variability, not single numbers: mean +/- std over seeds,
and confidence intervals for any headline metric. These helpers make that the
easy path.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample std (ddof=1) over seeds/runs; n is always reported."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1-dimensional sequence")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": float(len(arr)),
    }


def bootstrap_ci(
    metric_fn: Callable[..., float],
    predicted: Sequence[Any],
    reference: Sequence[Any],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for any pure metric of paired (pred, ref) data.

    Resamples example indices with replacement and recomputes ``metric_fn`` on
    each resample. Deterministic for a fixed ``seed``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if len(predicted) != len(reference):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(reference)}")
    n = len(predicted)
    if n == 0:
        raise ValueError("cannot bootstrap zero examples")
    pred_arr = np.asarray(predicted)
    ref_arr = np.asarray(reference)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(pred_arr[idx], ref_arr[idx])))
    lo = (1.0 - confidence) / 2.0
    return {
        "point": float(metric_fn(pred_arr, ref_arr)),
        "low": float(np.quantile(scores, lo)),
        "high": float(np.quantile(scores, 1.0 - lo)),
        "confidence": confidence,
        "n_resamples": float(n_resamples),
    }


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for mean difference of paired per-example scores.

    The standard system-comparison test on a shared test set: sign-flips the
    per-example differences. Returns the probability, under exchangeability, of
    a mean absolute difference at least as large as observed.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("scores must be non-empty 1-dimensional and aligned")
    diff = a - b
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diff)))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_resamples + 1))

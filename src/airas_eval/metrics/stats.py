"""Statistics over repeated runs and paired systems.

Papers must report variability, not single numbers. These are the two pieces
the evaluator needs: mean +/- sample std over seeds, and a paired sign-flip
permutation test for "A beats B on the same examples". Kept in-house because
both are a few lines with no ambiguous variant; the test is exact in its
Monte-Carlo sense and deterministic for a fixed seed.
"""

from collections.abc import Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample std (ddof=1; 0.0 for a single value), with n."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1-dimensional sequence")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": float(len(arr)),
    }


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for the mean of paired per-example differences.

    Sign-flips the differences ``a - b`` under the null of exchangeability and
    returns the fraction of resamples whose |mean| is at least the observed
    one (with the +1 correction so p is never 0).
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

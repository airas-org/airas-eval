"""Statistics over repeated runs and paired systems.

Papers must report variability, not single numbers. These are the two pieces
the evaluator needs: descriptive statistics over seeds (delegated to numpy and
scipy.stats — nothing is hand-rolled), and a paired sign-flip permutation test
for "A beats B on the same examples", which is exact in its Monte-Carlo sense
and deterministic for a fixed seed.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.stats


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """Descriptive statistics over repeated runs, all delegated to numpy/scipy.

    Returns mean, sample std (ddof=1), sem, min, max, median, q25, q75, a 95%
    t-interval (``ci95_low``/``ci95_high``), ``n`` and the raw ``values``.
    Dispersion (std, sem, ci95_*) is ``None`` for a single value — one run
    has no variability to report, and ``0.0`` would read as "no spread".
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1-dimensional sequence")
    n = len(arr)
    mean = float(np.mean(arr))
    out: dict[str, Any] = {
        "n": float(n),
        "mean": mean,
        "std": None,
        "sem": None,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "ci95_low": None,
        "ci95_high": None,
        "values": arr.tolist(),
    }
    if n > 1:
        sem = float(scipy.stats.sem(arr, ddof=1))
        if sem > 0.0:
            low, high = scipy.stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        else:  # identical values: the interval degenerates to the mean
            low = high = mean
        out.update(
            std=float(np.std(arr, ddof=1)),
            sem=sem,
            ci95_low=float(low),
            ci95_high=float(high),
        )
    return out


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

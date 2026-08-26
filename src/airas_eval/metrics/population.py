"""Position of a single result relative to a population of results.

For any setting where the space of possible outcomes is (partly) known — a
tabular NAS benchmark listing every architecture's accuracy, randomly
sampled architectures trained with the same pipeline, a leaderboard. These
put a raw score in context: how much of the space beats it, what random
sampling with the same budget would be expected to reach, how far above the
population mean it sits. Scores are higher-is-better.
"""

from collections.abc import Sequence

import numpy as np

from airas_eval.exceptions import UndefinedMetric


def _population(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("population must be a non-empty 1-dimensional sequence")
    return arr


def fraction_better(value: float, population: Sequence[float]) -> float:
    """Fraction of the population strictly better than ``value``.

    0.0 means nothing in the population beats it; 0.01 means it is in the
    top 1 %. Ties do not count as better.
    """
    return float(np.mean(_population(population) > float(value)))


def expected_best_of_random_sample(population: Sequence[float], n: int) -> float:
    """Expected maximum of ``n`` uniform draws with replacement from the population.

    Closed form from order statistics: with ``F`` the empirical CDF,
    ``E[max] = sum_v v * (F(v)^n - F(v-)^n)`` over distinct values. This is
    the random-search baseline at the same evaluation budget, without
    simulation noise.
    """
    arr = _population(population)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    values, counts = np.unique(arr, return_counts=True)
    cdf = np.cumsum(counts) / len(arr)
    cdf_prev = np.concatenate(([0.0], cdf[:-1]))
    return float(np.sum(values * (cdf**n - cdf_prev**n)))


def gain_over_random_search(
    best: float, population: Sequence[float], n_evaluations: int
) -> float:
    """``best - E[best of n random draws]``: what the search strategy added over
    random search with the same budget, in score units. Negative means random
    search would be expected to do better."""
    return float(best) - expected_best_of_random_sample(population, n_evaluations)


def relative_improvement(value: float, population: Sequence[float]) -> float:
    """``(value - mean(population)) / |mean(population)|``.

    Yang, Esperança & Carlucci (ICLR 2020): improvement relative to the
    average randomly sampled architecture, which factors out the search
    space and training protocol. Undefined when the population mean is 0.
    """
    mean = float(np.mean(_population(population)))
    if mean == 0.0:
        raise UndefinedMetric(
            "relative improvement is undefined for a zero-mean population"
        )
    return (float(value) - mean) / abs(mean)

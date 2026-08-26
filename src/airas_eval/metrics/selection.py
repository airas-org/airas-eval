"""Candidate-selection quality: how well predicted scores identify the truly
best candidates.

For performance predictors and zero-cost proxies in NAS, surrogate models in
HPO, or any scorer used to shortlist candidates. Complements the global rank
correlations in ``metrics.regression`` with top-of-ranking metrics (a
predictor is used to pick the top few, so global correlation alone can
mislead). Scores are higher-is-better; ranks are 1-based; ties are broken by
stable input order, so results are deterministic for identical inputs.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics import regression as _reg
from airas_eval.metrics._validate import paired_1d


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:k]


def _check_k(k: int, n: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise UndefinedMetric(f"k={k} exceeds the {n} candidates given")


def precision_at_top_fraction(
    predicted_scores: Sequence[float],
    reference_scores: Sequence[float],
    fraction: float,
) -> float:
    """Overlap of the predicted and true top-``fraction`` sets, over set size.

    ``k = floor(n * fraction)``; undefined when that set is empty.
    """
    pred, ref = paired_1d(predicted_scores, reference_scores, dtype=float)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = int(len(pred) * fraction)
    if k < 1:
        raise UndefinedMetric(
            f"the top {fraction:.0%} of {len(pred)} candidates is empty"
        )
    top_pred = set(_top_k_indices(pred, k).tolist())
    top_ref = set(_top_k_indices(ref, k).tolist())
    return len(top_pred & top_ref) / k


def best_true_rank_in_predicted_top_k(
    predicted_scores: Sequence[float],
    reference_scores: Sequence[float],
    k: int,
) -> float:
    """N@k: the best true rank (1-based) among the predictor's top-k picks.

    1.0 means the predictor's top-k contains the truly best candidate.
    """
    pred, ref = paired_1d(predicted_scores, reference_scores, dtype=float)
    _check_k(k, len(pred))
    true_rank = np.empty(len(ref), dtype=float)
    true_rank[np.argsort(-ref, kind="stable")] = np.arange(1, len(ref) + 1)
    return float(true_rank[_top_k_indices(pred, k)].min())


def selection_regret_at_k(
    predicted_scores: Sequence[float],
    reference_scores: Sequence[float],
    k: int,
) -> float:
    """True-score gap between the overall best candidate and the best among
    the predictor's top-k picks: the cost of trusting the predictor."""
    pred, ref = paired_1d(predicted_scores, reference_scores, dtype=float)
    _check_k(k, len(pred))
    return float(ref.max() - ref[_top_k_indices(pred, k)].max())


def rank_correlation_top_fraction(
    predicted_scores: Sequence[float],
    reference_scores: Sequence[float],
    fraction: float,
    method: Literal["kendall", "spearman"],
) -> float:
    """Rank correlation restricted to the candidates whose TRUE score is in the
    top ``fraction``: does the predictor still order the good ones correctly?

    Standard alongside global correlation for zero-cost proxies and
    performance predictors (NAS-Bench-Suite-Zero). Undefined when the top set
    has fewer than two candidates or a constant input.
    """
    pred, ref = paired_1d(predicted_scores, reference_scores, dtype=float)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = int(len(ref) * fraction)
    if k < 2:
        raise UndefinedMetric(
            f"the top {fraction:.0%} of {len(ref)} candidates has fewer than 2 members"
        )
    top = _top_k_indices(ref, k)
    fn = _reg.kendall_tau if method == "kendall" else _reg.spearman_rho
    return fn(pred[top], ref[top])

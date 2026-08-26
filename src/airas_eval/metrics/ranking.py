"""Ranking / retrieval metrics as pure functions.

Convention: each query has a ranked list of item ids (best first) and a set of
relevant item ids (or graded relevances for nDCG). Corpus-level metrics are the
mean over queries.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not ranked:
        raise ValueError("ranked list must be non-empty")
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        raise ValueError("relevant set must be non-empty")
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: Sequence[Sequence[str]], relevant_sets: Sequence[set[str]]
) -> float:
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must align")
    if not ranked_lists:
        raise ValueError("need at least one query")
    return float(
        np.mean(
            [
                reciprocal_rank(r, s)
                for r, s in zip(ranked_lists, relevant_sets, strict=False)
            ]
        )
    )


def average_precision_at_k(
    ranked: Sequence[str], relevant: set[str], k: int | None = None
) -> float:
    """AP for one query: mean of precision@i over ranks i that hit a relevant item."""
    if not relevant:
        raise ValueError("relevant set must be non-empty")
    limit = len(ranked) if k is None else min(k, len(ranked))
    hits = 0
    precision_sum = 0.0
    for i in range(limit):
        if ranked[i] in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    denom = min(len(relevant), limit)
    return precision_sum / denom if denom else 0.0


def mean_average_precision(
    ranked_lists: Sequence[Sequence[str]],
    relevant_sets: Sequence[set[str]],
    k: int | None = None,
) -> float:
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must align")
    if not ranked_lists:
        raise ValueError("need at least one query")
    return float(
        np.mean(
            [
                average_precision_at_k(r, s, k)
                for r, s in zip(ranked_lists, relevant_sets, strict=False)
            ]
        )
    )


def ndcg_at_k(ranked: Sequence[str], relevances: Mapping[str, float], k: int) -> float:
    """nDCG@k with graded relevance and log2 discount (standard formulation)."""
    _check_k(k)
    gains = [relevances.get(item, 0.0) for item in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevances.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        raise ValueError("nDCG is undefined when no item has positive relevance")
    return float(dcg / idcg)

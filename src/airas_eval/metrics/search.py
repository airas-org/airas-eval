"""Search / optimization outcome metrics.

For any process that evaluates candidates in sequence and keeps the best —
neural architecture search on a tabular benchmark being the primary AIRAS
use case, but nothing here is NAS-specific. Scores are higher-is-better;
``oracle_best`` is the published optimum of the benchmark being searched.
"""

from collections.abc import Sequence

import numpy as np


def _scores_1d(scores: Sequence[float], name: str = "scores") -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1-dimensional sequence")
    return arr


def best_so_far(evaluated_scores: Sequence[float]) -> list[float]:
    """Incumbent (running best) score after each evaluation, in order."""
    incumbent = np.maximum.accumulate(_scores_1d(evaluated_scores))
    return [float(v) for v in incumbent]


def best_score(evaluated_scores: Sequence[float]) -> float:
    """Best score found over the whole evaluation budget."""
    return float(_scores_1d(evaluated_scores).max())


def evaluations_to_best(evaluated_scores: Sequence[float]) -> float:
    """1-based index of the evaluation that first reached the final best score."""
    arr = _scores_1d(evaluated_scores)
    return float(int(np.argmax(arr)) + 1)


def mean_evaluated_score(evaluated_scores: Sequence[float]) -> float:
    """Mean score of all evaluated candidates: the quality of what the search
    chose to evaluate, as opposed to the best it happened to find."""
    return float(_scores_1d(evaluated_scores).mean())


def _costs_1d(
    evaluated_scores: Sequence[float], evaluation_costs: Sequence[float]
) -> np.ndarray:
    scores = _scores_1d(evaluated_scores)
    costs = _scores_1d(evaluation_costs, "evaluation_costs")
    if len(costs) != len(scores):
        raise ValueError(f"length mismatch: {len(scores)} scores vs {len(costs)} costs")
    if np.any(costs < 0):
        raise ValueError("evaluation_costs must be non-negative")
    return costs


def total_cost(
    evaluated_scores: Sequence[float], evaluation_costs: Sequence[float]
) -> float:
    """Sum of per-evaluation costs: the estimated wall-clock budget consumed."""
    return float(_costs_1d(evaluated_scores, evaluation_costs).sum())


def cost_to_best(
    evaluated_scores: Sequence[float], evaluation_costs: Sequence[float]
) -> float:
    """Cumulative cost spent when the final best score was first reached.

    The cost-axis counterpart of ``evaluations_to_best``: NAS benchmarks plot
    the incumbent against estimated training time, not against the number
    of architectures evaluated.
    """
    costs = _costs_1d(evaluated_scores, evaluation_costs)
    idx = int(np.argmax(np.asarray(evaluated_scores, dtype=float)))
    return float(np.cumsum(costs)[idx])


def best_so_far_vs_cost(
    evaluated_scores: Sequence[float], evaluation_costs: Sequence[float]
) -> list[list[float]]:
    """``[[cumulative_cost, incumbent], ...]`` after each evaluation."""
    costs = _costs_1d(evaluated_scores, evaluation_costs)
    incumbent = best_so_far(evaluated_scores)
    return [[float(c), v] for c, v in zip(np.cumsum(costs), incumbent, strict=True)]


def _check_oracle(best: float, oracle_best: float) -> None:
    # A score above the known optimum means the inputs are inconsistent with
    # the benchmark. That is an integrity failure of the whole evaluation,
    # not an undefined metric — raise ValueError so the evaluator fails loudly
    # instead of quietly dropping the regret column.
    if best > oracle_best:
        raise ValueError(
            f"best evaluated score {best} exceeds the benchmark optimum "
            f"{oracle_best}; inputs are inconsistent with the benchmark"
        )


def final_regret(evaluated_scores: Sequence[float], oracle_best: float) -> float:
    """``oracle_best - best_score``: distance to the benchmark optimum."""
    best = best_score(evaluated_scores)
    _check_oracle(best, oracle_best)
    return float(oracle_best) - best


def mean_anytime_regret(evaluated_scores: Sequence[float], oracle_best: float) -> float:
    """Mean of ``oracle_best - best_so_far`` over the evaluation budget.

    Summarizes the anytime curve: low values mean good candidates were found
    early, not just eventually.
    """
    incumbent = np.asarray(best_so_far(evaluated_scores))
    _check_oracle(float(incumbent[-1]), oracle_best)
    return float(np.mean(oracle_best - incumbent))

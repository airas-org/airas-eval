"""NAS bundles = the core bundle's bindings + NAS-specific ones.

The core part is *the same tuple object* as the generic bundle uses, so a
``nas_search`` report contains everything a ``search`` report does, plus
what the NAS literature adds: the estimated-wall-clock axis (NAS-Bench-101 /
201), position in the search space and a closed-form random-search baseline
at equal budget (Lindauer & Hutter 2020; Yang et al. 2020), validation vs
test regret, and top-10 % rank correlation for predictors
(NAS-Bench-Suite-Zero).
"""

from airas_eval.metrics import classification as _cls
from airas_eval.metrics import population as _pop
from airas_eval.metrics import search as _search
from airas_eval.metrics import selection as _sel
from airas_eval.spec import Bundle, MetricBinding
from airas_eval.tasks import _bundles as core
from airas_eval.tasks.nas._inputs import NasArchitectureInputs, NasSearchInputs

# --- search --------------------------------------------------------------------

_SCORES = ("evaluated_scores",)
_WITH_COSTS = ("evaluated_scores", "evaluation_costs")
_WITH_SPACE = ("evaluated_scores", "search_space_scores")


def search_space_fraction_better(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.fraction_better(
        _search.best_score(evaluated_scores), search_space_scores
    )


def gain_over_random_search(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.gain_over_random_search(
        _search.best_score(evaluated_scores), search_space_scores, len(evaluated_scores)
    )


def relative_improvement_over_random(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.relative_improvement(
        _search.best_score(evaluated_scores), search_space_scores
    )


def test_regret(final_test_score: float, oracle_test_best: float) -> float:
    return _search.final_regret([final_test_score], oracle_test_best)


def n_search_space(search_space_scores: list[float]) -> float:
    return float(len(search_space_scores))


SEARCH = Bundle(
    input_model=NasSearchInputs,
    provenance_packages=core.SEARCH.provenance_packages,
    notes=(
        core.SEARCH.notes + "; costs are the benchmark's per-architecture training "
        "cost, cumulated in evaluation order; the random-search baseline is the "
        "exact expected best of n uniform draws from search_space_scores; "
        "relative improvement is against the search-space mean (Yang et al. 2020)"
    ),
    metrics=core.SEARCH.metrics
    + (
        MetricBinding("cost_to_best", _search.cost_to_best, _WITH_COSTS),
        MetricBinding(
            "search_space_fraction_better", search_space_fraction_better, _WITH_SPACE
        ),
        MetricBinding("gain_over_random_search", gain_over_random_search, _WITH_SPACE),
        MetricBinding(
            "relative_improvement_over_random",
            relative_improvement_over_random,
            _WITH_SPACE,
        ),
        MetricBinding(
            "test_regret", test_regret, ("final_test_score", "oracle_test_best")
        ),
    ),
    curves=core.SEARCH.curves
    + (MetricBinding("best_so_far_vs_cost", _search.best_so_far_vs_cost, _WITH_COSTS),),
    summary=core.SEARCH.summary
    + (
        MetricBinding("total_cost", _search.total_cost, _WITH_COSTS),
        MetricBinding("n_search_space", n_search_space, ("search_space_scores",)),
    ),
)

# --- predictor -----------------------------------------------------------------

_PAIR = ("predicted_scores", "reference_scores")

PREDICTOR = Bundle(
    input_model=core.CANDIDATE_RANKING.input_model,
    provenance_packages=core.CANDIDATE_RANKING.provenance_packages,
    notes=(
        core.CANDIDATE_RANKING.notes
        + "; top-10% correlations are computed over the candidates whose TRUE "
        "score is in the top 10% (NAS-Bench-Suite-Zero protocol)"
    ),
    metrics=core.CANDIDATE_RANKING.metrics
    + (
        MetricBinding(
            "kendall_tau_top_10pct",
            _sel.rank_correlation_top_fraction,
            _PAIR,
            {"fraction": 0.10, "method": "kendall"},
        ),
        MetricBinding(
            "spearman_rho_top_10pct",
            _sel.rank_correlation_top_fraction,
            _PAIR,
            {"fraction": 0.10, "method": "spearman"},
        ),
    ),
    summary=core.CANDIDATE_RANKING.summary,
)

# --- architecture --------------------------------------------------------------

_LABELS = ("predicted_labels", "reference_labels")
_VS_RANDOM = ("predicted_labels", "reference_labels", "random_architecture_accuracies")


def relative_improvement_over_random_architectures(
    predicted_labels: list[int],
    reference_labels: list[int],
    random_architecture_accuracies: list[float],
) -> float:
    return _pop.relative_improvement(
        _cls.accuracy(predicted_labels, reference_labels),
        random_architecture_accuracies,
    )


def fraction_of_random_architectures_better(
    predicted_labels: list[int],
    reference_labels: list[int],
    random_architecture_accuracies: list[float],
) -> float:
    return _pop.fraction_better(
        _cls.accuracy(predicted_labels, reference_labels),
        random_architecture_accuracies,
    )


def n_random_architectures(random_architecture_accuracies: list[float]) -> float:
    return float(len(random_architecture_accuracies))


ARCHITECTURE = Bundle(
    input_model=NasArchitectureInputs,
    provenance_packages=core.CLASSIFICATION.provenance_packages,
    notes=(
        core.CLASSIFICATION.notes
        + "; the random-architecture baseline compares top-1 accuracy against "
        "architectures sampled uniformly from the same search space and trained "
        "with the same pipeline (Yang et al. 2020)"
    ),
    metrics=core.CLASSIFICATION.metrics
    + (
        MetricBinding(
            "relative_improvement_over_random",
            relative_improvement_over_random_architectures,
            _VS_RANDOM,
        ),
        MetricBinding(
            "fraction_of_random_better",
            fraction_of_random_architectures_better,
            _VS_RANDOM,
        ),
    ),
    summary=core.CLASSIFICATION.summary
    + (
        MetricBinding(
            "n_random_architectures",
            n_random_architectures,
            ("random_architecture_accuracies",),
        ),
    ),
)

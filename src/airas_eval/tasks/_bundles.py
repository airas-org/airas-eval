"""Reusable metric bundles: the building blocks tasks are composed from.

A bundle is "all the standard metrics over one input shape". Bundles are
module constants — not registered, not evaluable on their own — so a caller
can only ever pick a task type, never a subset of it. Adapters that need
task-shape knowledge (binary-only metrics, top-5 needing > 5 classes, input
counts) live here as named module-level functions so they carry a stable
qualname into the task signature.
"""

import numpy as np

from airas_eval.exceptions import NotApplicable, UndefinedMetric
from airas_eval.metrics import classification as _cls
from airas_eval.metrics import pareto as _pareto
from airas_eval.metrics import regression as _reg
from airas_eval.metrics import search as _search
from airas_eval.metrics import selection as _sel
from airas_eval.spec import Bundle, MetricBinding
from airas_eval.tasks._inputs import (
    CandidateRankingInputs,
    ClassificationInputs,
    MultiobjectiveInputs,
    SearchInputs,
)

Probs = list[list[float]]
Labels = list[int]

# --- classification -----------------------------------------------------------


def _binary_positive_scores(
    probabilities: Probs, reference_labels: Labels
) -> tuple[list[float], list[int]]:
    probs = np.asarray(probabilities, dtype=float)
    reference = np.asarray(reference_labels)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise NotApplicable("binary-only metric: probabilities are not 2-class")
    if set(np.unique(reference).tolist()) != {0, 1}:
        raise UndefinedMetric("reference labels do not contain both classes 0 and 1")
    return probs[:, 1].tolist(), reference.tolist()


def auroc_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.auroc(*_binary_positive_scores(probabilities, reference_labels))


def average_precision_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.average_precision(
        *_binary_positive_scores(probabilities, reference_labels)
    )


def brier_score_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.brier_score(*_binary_positive_scores(probabilities, reference_labels))


def n_examples(predicted_labels: Labels, reference_labels: Labels) -> float:
    if len(predicted_labels) != len(reference_labels):
        raise ValueError(
            f"length mismatch: {len(predicted_labels)} vs {len(reference_labels)}"
        )
    return float(len(reference_labels))


def n_classes(probabilities: Probs) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-dimensional, got {probs.shape}")
    return float(probs.shape[1])


def top_5_accuracy(probabilities: Probs, reference_labels: Labels) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] <= 5:
        raise NotApplicable("top-5 accuracy requires more than 5 classes")
    return _cls.top_k_accuracy(probabilities, reference_labels, k=5)


_LABELS = ("predicted_labels", "reference_labels")
_PROBS = ("probabilities", "reference_labels")

CLASSIFICATION = Bundle(
    input_model=ClassificationInputs,
    provenance_packages=("numpy", "scikit-learn"),
    notes=(
        "single-label multiclass; precision/recall/F1 are macro-averaged with "
        "zero_division=0 (micro equals accuracy here); ECE uses 15 equal-width "
        "bins over top-1 confidence"
    ),
    metrics=(
        MetricBinding("accuracy", _cls.accuracy, _LABELS),
        MetricBinding("precision_macro", _cls.precision, _LABELS, {"average": "macro"}),
        MetricBinding("recall_macro", _cls.recall, _LABELS, {"average": "macro"}),
        MetricBinding("f1_macro", _cls.f1, _LABELS, {"average": "macro"}),
        MetricBinding("balanced_accuracy", _cls.balanced_accuracy, _LABELS),
        MetricBinding("matthews_corrcoef", _cls.matthews_corrcoef, _LABELS),
        MetricBinding("log_loss", _cls.log_loss, _PROBS),
        MetricBinding(
            "expected_calibration_error",
            _cls.expected_calibration_error,
            _PROBS,
            {"n_bins": 15},
        ),
        MetricBinding("top_5_accuracy", top_5_accuracy, _PROBS),
    ),
    summary=(
        MetricBinding("n_examples", n_examples, _LABELS),
        MetricBinding("n_classes", n_classes, ("probabilities",)),
    ),
)

BINARY_CLASSIFICATION = Bundle(
    input_model=ClassificationInputs,
    provenance_packages=("numpy", "scikit-learn"),
    notes=(
        "labels in {0, 1}; score metrics read the positive class from "
        "probabilities[:, 1]; F1 is for the positive class; ECE uses 15 "
        "equal-width bins over top-1 confidence"
    ),
    metrics=(
        MetricBinding("accuracy", _cls.accuracy, _LABELS),
        MetricBinding("precision", _cls.precision, _LABELS, {"average": "binary"}),
        MetricBinding("recall", _cls.recall, _LABELS, {"average": "binary"}),
        MetricBinding("f1", _cls.f1, _LABELS, {"average": "binary"}),
        MetricBinding("balanced_accuracy", _cls.balanced_accuracy, _LABELS),
        MetricBinding("matthews_corrcoef", _cls.matthews_corrcoef, _LABELS),
        MetricBinding("auroc", auroc_binary, _PROBS),
        MetricBinding("average_precision", average_precision_binary, _PROBS),
        MetricBinding("log_loss", _cls.log_loss, _PROBS),
        MetricBinding("brier_score", brier_score_binary, _PROBS),
        MetricBinding(
            "expected_calibration_error",
            _cls.expected_calibration_error,
            _PROBS,
            {"n_bins": 15},
        ),
    ),
    summary=(MetricBinding("n_examples", n_examples, _LABELS),),
)

# --- search -------------------------------------------------------------------


def n_evaluations(evaluated_scores: list[float]) -> float:
    return float(len(evaluated_scores))


_SCORES = ("evaluated_scores",)
_WITH_ORACLE = ("evaluated_scores", "oracle_best")

SEARCH = Bundle(
    input_model=SearchInputs,
    provenance_packages=("numpy",),
    notes=(
        "scores are higher-is-better in evaluation order; oracle_best is "
        "fixed by the experimental design; regret fails, not skips, if a "
        "score exceeds it"
    ),
    metrics=(
        MetricBinding("best_score", _search.best_score, _SCORES),
        MetricBinding("final_regret", _search.final_regret, _WITH_ORACLE),
        MetricBinding("mean_anytime_regret", _search.mean_anytime_regret, _WITH_ORACLE),
        MetricBinding("evaluations_to_best", _search.evaluations_to_best, _SCORES),
        MetricBinding("mean_evaluated_score", _search.mean_evaluated_score, _SCORES),
    ),
    curves=(MetricBinding("best_so_far", _search.best_so_far, _SCORES),),
    summary=(MetricBinding("n_evaluations", n_evaluations, _SCORES),),
)

# --- candidate ranking --------------------------------------------------------


def n_candidates(predicted_scores: list[float], reference_scores: list[float]) -> float:
    if len(predicted_scores) != len(reference_scores):
        raise ValueError(
            f"length mismatch: {len(predicted_scores)} vs {len(reference_scores)}"
        )
    return float(len(reference_scores))


_PAIR = ("predicted_scores", "reference_scores")

# TODO(k-sweep): callers only see the report (uvx airas-eval score), so the
# way to expose "what if k / fraction were different" is a curve, not a knob:
# add ``selection_regret_curve`` (regret@k for k=1..n) and
# ``precision_at_top_k_curve`` under ``curves``. The scalar pins stay as the
# summary values.
CANDIDATE_RANKING = Bundle(
    input_model=CandidateRankingInputs,
    provenance_packages=("numpy", "scipy"),
    notes=(
        "scores are higher-is-better; Kendall is tau-b; top-k sets use "
        "stable descending order for ties; ranks are 1-based"
    ),
    metrics=(
        MetricBinding("kendall_tau", _reg.kendall_tau, _PAIR),
        MetricBinding("spearman_rho", _reg.spearman_rho, _PAIR),
        MetricBinding(
            "precision_at_top_10pct",
            _sel.precision_at_top_fraction,
            _PAIR,
            {"fraction": 0.10},
        ),
        MetricBinding(
            "selection_regret_at_1", _sel.selection_regret_at_k, _PAIR, {"k": 1}
        ),
        MetricBinding(
            "best_true_rank_in_top_10",
            _sel.best_true_rank_in_predicted_top_k,
            _PAIR,
            {"k": 10},
        ),
    ),
    summary=(MetricBinding("n_candidates", n_candidates, _PAIR),),
)

# --- multiobjective -----------------------------------------------------------


def n_points(points: list[list[float]]) -> float:
    return float(len(points))


def n_objectives(points: list[list[float]]) -> float:
    widths = {len(row) for row in points}
    if len(widths) != 1:
        raise ValueError("points must all have the same number of objectives")
    return float(widths.pop())


def pareto_front_size(points: list[list[float]]) -> float:
    return float(sum(_pareto.pareto_front_mask(points)))


_POINTS = ("points",)

MULTIOBJECTIVE = Bundle(
    input_model=MultiobjectiveInputs,
    provenance_packages=("numpy",),
    notes=(
        "all objectives are minimized; hypervolume is exact 2-D; IGD/GD/spacing "
        "are unnormalized Euclidean (normalize objectives before calling)"
    ),
    metrics=(
        MetricBinding("pareto_front_size", pareto_front_size, _POINTS),
        MetricBinding(
            "hypervolume_2d", _pareto.hypervolume_2d, ("points", "reference_point")
        ),
        MetricBinding("igd", _pareto.igd, ("points", "reference_front")),
        MetricBinding("gd", _pareto.gd, ("points", "reference_front")),
        MetricBinding("spacing", _pareto.spacing, _POINTS),
    ),
    curves=(MetricBinding("pareto_front", _pareto.pareto_front, _POINTS),),
    summary=(
        MetricBinding("n_points", n_points, _POINTS),
        MetricBinding("n_objectives", n_objectives, _POINTS),
    ),
)

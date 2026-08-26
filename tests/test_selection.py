import pytest

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics import selection


def test_precision_at_top_fraction_perfect_and_disjoint():
    scores = list(range(10))
    assert selection.precision_at_top_fraction(scores, scores, 0.2) == 1.0
    assert selection.precision_at_top_fraction(scores, scores[::-1], 0.2) == 0.0


def test_precision_at_top_fraction_partial_overlap():
    predicted = [10, 9, 1, 2, 8, 3, 4, 5, 6, 7]
    reference = [10, 8, 1, 2, 9, 3, 4, 5, 6, 7]
    # top-2 by prediction: indices {0, 1}; by reference: {0, 4} -> overlap 1/2
    assert selection.precision_at_top_fraction(predicted, reference, 0.2) == 0.5


def test_precision_at_top_fraction_undefined_and_invalid():
    with pytest.raises(UndefinedMetric):
        selection.precision_at_top_fraction([1, 2, 3], [1, 2, 3], 0.1)
    with pytest.raises(ValueError, match="fraction"):
        selection.precision_at_top_fraction([1, 2, 3], [1, 2, 3], 1.5)


def test_best_true_rank_in_predicted_top_k():
    reference = [1.0, 5.0, 3.0, 4.0]  # true best is index 1
    predicted_good = [0.0, 9.0, 1.0, 2.0]  # ranks index 1 first
    predicted_bad = [9.0, 0.0, 5.0, 6.0]  # ranks index 0 first
    assert (
        selection.best_true_rank_in_predicted_top_k(predicted_good, reference, 1) == 1.0
    )
    # predictor's top-1 is index 0, whose true rank is 4
    assert (
        selection.best_true_rank_in_predicted_top_k(predicted_bad, reference, 1) == 4.0
    )
    with pytest.raises(UndefinedMetric):
        selection.best_true_rank_in_predicted_top_k(reference, reference, 10)


def test_selection_regret_at_k():
    reference = [1.0, 5.0, 3.0, 4.0]
    perfect = [1.0, 5.0, 3.0, 4.0]
    assert selection.selection_regret_at_k(perfect, reference, 1) == 0.0
    misleading = [9.0, 0.0, 5.0, 6.0]  # picks index 0 (true score 1.0)
    assert selection.selection_regret_at_k(misleading, reference, 1) == pytest.approx(
        4.0
    )
    # at k=3 the picks are indices {0, 3, 2}: best true score 4.0 -> regret 1
    assert selection.selection_regret_at_k(misleading, reference, 3) == pytest.approx(
        1.0
    )

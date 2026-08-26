import pytest

from airas_eval.metrics import search


def test_best_so_far_and_best_score():
    assert search.best_so_far([1.0, 3.0, 2.0, 5.0]) == [1.0, 3.0, 3.0, 5.0]
    assert search.best_score([1.0, 3.0, 2.0]) == 3.0


def test_final_regret():
    assert search.final_regret([1.0, 3.0, 2.0], oracle_best=5.0) == pytest.approx(2.0)
    assert search.final_regret([5.0], oracle_best=5.0) == 0.0


def test_regret_fails_when_score_exceeds_oracle():
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        search.final_regret([5.1], oracle_best=5.0)
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        search.mean_anytime_regret([5.1], oracle_best=5.0)


def test_mean_anytime_regret():
    # best_so_far = [1, 3, 3] -> regrets [4, 2, 2]
    assert search.mean_anytime_regret(
        [1.0, 3.0, 2.0], oracle_best=5.0
    ) == pytest.approx(8.0 / 3.0)


def test_empty_scores_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        search.best_score([])


def test_evaluations_to_best_and_mean_score():
    assert search.evaluations_to_best([1.0, 3.0, 2.0, 3.0]) == 2.0  # first hit
    assert search.evaluations_to_best([5.0]) == 1.0
    assert search.mean_evaluated_score([1.0, 3.0, 2.0]) == pytest.approx(2.0)

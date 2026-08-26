import pytest

from airas_eval.metrics import ranking as rk

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d", "x"}


def test_precision_at_k():
    assert rk.precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    assert rk.precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_recall_at_k():
    assert rk.recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert rk.recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_hit_rate():
    assert rk.hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
    assert rk.hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0


def test_reciprocal_rank():
    assert rk.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    assert rk.reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_mrr():
    assert rk.mean_reciprocal_rank(
        [RANKED, ["x", "a"]], [RELEVANT, {"x"}]
    ) == pytest.approx((0.5 + 1.0) / 2)


def test_average_precision_hand_computed():
    # Hits at ranks 2 and 4: AP = (1/2 + 2/4) / min(3, 5) = 1/3
    assert rk.average_precision_at_k(RANKED, RELEVANT) == pytest.approx(1 / 3)


def test_map():
    assert rk.mean_average_precision([RANKED], [RELEVANT]) == pytest.approx(1 / 3)


def test_ndcg_perfect_ranking_is_one():
    relevances = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert rk.ndcg_at_k(["a", "b", "c"], relevances, 3) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    import numpy as np

    relevances = {"a": 1.0, "b": 1.0}
    # Ranking ["x", "a", "b"]: DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1.0 + 1 / np.log2(3)
    assert rk.ndcg_at_k(["x", "a", "b"], relevances, 3) == pytest.approx(dcg / idcg)


def test_ndcg_no_positive_relevance_raises():
    with pytest.raises(ValueError):
        rk.ndcg_at_k(["a"], {}, 1)

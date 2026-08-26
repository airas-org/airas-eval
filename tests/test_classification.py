import numpy as np
import pytest
from sklearn import metrics as skm

from airas_eval.metrics import classification as c

rng = np.random.default_rng(42)
N = 500
N_CLASSES = 4
Y_TRUE = rng.integers(0, N_CLASSES, size=N)
Y_PRED = np.where(rng.random(N) < 0.7, Y_TRUE, rng.integers(0, N_CLASSES, size=N))
PROBS = rng.dirichlet(np.ones(N_CLASSES), size=N)
Y_BIN = rng.integers(0, 2, size=N)
SCORES = np.clip(Y_BIN * 0.4 + rng.random(N) * 0.6, 0, 1)


def test_accuracy_matches_sklearn():
    assert c.accuracy(Y_PRED, Y_TRUE) == pytest.approx(
        skm.accuracy_score(Y_TRUE, Y_PRED)
    )


def test_error_rate_is_complement():
    assert c.error_rate(Y_PRED, Y_TRUE) == pytest.approx(
        1 - skm.accuracy_score(Y_TRUE, Y_PRED)
    )


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_precision_recall_f1_match_sklearn(average):
    assert c.precision(Y_PRED, Y_TRUE, average) == pytest.approx(
        skm.precision_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
    )
    assert c.recall(Y_PRED, Y_TRUE, average) == pytest.approx(
        skm.recall_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
    )
    assert c.f1(Y_PRED, Y_TRUE, average) == pytest.approx(
        skm.f1_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
    )


def test_balanced_accuracy_matches_sklearn():
    assert c.balanced_accuracy(Y_PRED, Y_TRUE) == pytest.approx(
        skm.balanced_accuracy_score(Y_TRUE, Y_PRED)
    )


def test_mcc_matches_sklearn():
    assert c.matthews_corrcoef(Y_PRED, Y_TRUE) == pytest.approx(
        skm.matthews_corrcoef(Y_TRUE, Y_PRED)
    )


def test_cohen_kappa_matches_sklearn():
    assert c.cohen_kappa(Y_PRED, Y_TRUE) == pytest.approx(
        skm.cohen_kappa_score(Y_TRUE, Y_PRED)
    )


def test_auroc_matches_sklearn():
    assert c.auroc(SCORES, Y_BIN) == pytest.approx(skm.roc_auc_score(Y_BIN, SCORES))


def test_auroc_with_ties_matches_sklearn():
    tied = np.round(SCORES, 1)
    assert c.auroc(tied, Y_BIN) == pytest.approx(skm.roc_auc_score(Y_BIN, tied))


def test_average_precision_matches_sklearn():
    assert c.average_precision(SCORES, Y_BIN) == pytest.approx(
        skm.average_precision_score(Y_BIN, SCORES)
    )


def test_log_loss_matches_sklearn():
    assert c.log_loss(PROBS, Y_TRUE) == pytest.approx(
        skm.log_loss(Y_TRUE, PROBS, labels=list(range(N_CLASSES)))
    )


def test_brier_matches_sklearn():
    assert c.brier_score(SCORES, Y_BIN) == pytest.approx(
        skm.brier_score_loss(Y_BIN, SCORES)
    )


def test_top_k_matches_sklearn():
    assert c.top_k_accuracy(PROBS, Y_TRUE, k=2) == pytest.approx(
        skm.top_k_accuracy_score(Y_TRUE, PROBS, k=2, labels=list(range(N_CLASSES)))
    )


def test_ece_zero_for_perfectly_calibrated_confident():
    probs = np.eye(3)[[0, 1, 2, 0]]
    labels = [0, 1, 2, 0]
    assert c.expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        c.accuracy([0, 1], [0])


def test_empty_raises():
    with pytest.raises(ValueError):
        c.accuracy([], [])

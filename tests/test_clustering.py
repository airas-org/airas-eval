import numpy as np
import pytest
from sklearn import metrics as skm

from airas_eval.metrics import clustering as cl

rng = np.random.default_rng(3)
N = 400
LABELS_TRUE = rng.integers(0, 5, size=N)
LABELS_PRED = np.where(rng.random(N) < 0.6, LABELS_TRUE, rng.integers(0, 6, size=N))


def test_ari_matches_sklearn():
    assert cl.adjusted_rand_index(LABELS_PRED, LABELS_TRUE) == pytest.approx(
        skm.adjusted_rand_score(LABELS_TRUE, LABELS_PRED)
    )


def test_nmi_matches_sklearn():
    assert cl.normalized_mutual_info(LABELS_PRED, LABELS_TRUE) == pytest.approx(
        skm.normalized_mutual_info_score(LABELS_TRUE, LABELS_PRED)
    )


def test_v_measure_matches_sklearn():
    assert cl.v_measure(LABELS_PRED, LABELS_TRUE) == pytest.approx(
        skm.v_measure_score(LABELS_TRUE, LABELS_PRED)
    )


def test_permutation_invariance():
    permuted = (LABELS_PRED + 3) % 7
    assert cl.adjusted_rand_index(permuted, LABELS_TRUE) == pytest.approx(
        cl.adjusted_rand_index(LABELS_PRED, LABELS_TRUE)
    )


def test_identical_labelings_score_one():
    assert cl.adjusted_rand_index(LABELS_TRUE, LABELS_TRUE) == pytest.approx(1.0)
    assert cl.normalized_mutual_info(LABELS_TRUE, LABELS_TRUE) == pytest.approx(1.0)
    assert cl.v_measure(LABELS_TRUE, LABELS_TRUE) == pytest.approx(1.0)


def test_ami_matches_sklearn():
    small_pred = LABELS_PRED[:120]
    small_true = LABELS_TRUE[:120]
    assert cl.adjusted_mutual_info(small_pred, small_true) == pytest.approx(
        skm.adjusted_mutual_info_score(small_true, small_pred)
    )

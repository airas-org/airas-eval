import numpy as np
import pytest
from sklearn import metrics as skm

from airas_eval.metrics import vision as v

PRED = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 1]])
TRUE = np.array([[0, 1, 0], [0, 2, 1], [0, 0, 1]])


def test_pixel_accuracy():
    assert v.pixel_accuracy(PRED, TRUE) == pytest.approx(7 / 9)


def test_binary_iou_matches_sklearn_jaccard():
    pred_bin = (PRED > 0).astype(int)
    true_bin = (TRUE > 0).astype(int)
    assert v.binary_iou(pred_bin, true_bin) == pytest.approx(
        skm.jaccard_score(true_bin.ravel(), pred_bin.ravel())
    )


def test_dice_from_iou_identity():
    iou = v.binary_iou(PRED, TRUE)
    dice = v.dice_coefficient(PRED, TRUE)
    assert dice == pytest.approx(2 * iou / (1 + iou))


def test_mean_iou_hand_computed():
    # class 0: inter 4, union 5; class 1: inter 2, union 4; class 2: inter 1, union 2
    expected = np.mean([4 / 5, 2 / 4, 1 / 2])
    assert v.mean_iou(PRED, TRUE) == pytest.approx(expected)


def test_psnr_hand_computed():
    pred = np.array([0.0, 0.5])
    true = np.array([0.0, 0.0])
    assert v.psnr(pred, true, data_range=1.0) == pytest.approx(
        10 * np.log10(1.0 / 0.125)
    )


def test_psnr_identical_raises():
    with pytest.raises(ValueError):
        v.psnr(TRUE.astype(float), TRUE.astype(float))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        v.pixel_accuracy(PRED, TRUE[:2])

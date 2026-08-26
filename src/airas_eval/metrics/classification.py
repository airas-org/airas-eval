"""Classification metrics, delegated to scikit-learn.

scikit-learn is the community-canonical implementation for these metrics;
airas-eval adds fail-closed input validation and pins the ambiguous variants
(averaging mode is an explicit argument; the suite layer reports all of them).
The only in-house computation is ECE, which has no canonical torch-free
implementation.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn import metrics as _skm

from airas_eval.metrics._validate import binary_scores, paired_1d, probs_2d

Average = Literal["micro", "macro", "weighted"]


def accuracy(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.accuracy_score(y_true, y_pred))


def error_rate(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return 1.0 - accuracy(predicted, reference)


def top_k_accuracy(
    probabilities: Sequence[Sequence[float]], reference: Sequence[int], k: int
) -> float:
    probs, y_true = probs_2d(probabilities, reference)
    if not 1 <= k <= probs.shape[1]:
        raise ValueError(f"k={k} out of range for {probs.shape[1]} classes")
    return float(
        _skm.top_k_accuracy_score(
            y_true, probs, k=k, labels=list(range(probs.shape[1]))
        )
    )


def precision(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.f1_score(y_true, y_pred, average=average, zero_division=0))


def macro_f1(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return f1(predicted, reference, average="macro")


def balanced_accuracy(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.balanced_accuracy_score(y_true, y_pred))


def matthews_corrcoef(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.matthews_corrcoef(y_true, y_pred))


def cohen_kappa(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.cohen_kappa_score(y_true, y_pred))


def auroc(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Binary ROC AUC on positive-class scores."""
    y_score, y_true = binary_scores(scores, reference)
    return float(_skm.roc_auc_score(y_true, y_score))


def average_precision(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Binary area under the precision-recall curve (AP, not trapezoidal)."""
    y_score, y_true = binary_scores(scores, reference)
    return float(_skm.average_precision_score(y_true, y_score))


def log_loss(
    probabilities: Sequence[Sequence[float]], reference: Sequence[int]
) -> float:
    probs, y_true = probs_2d(probabilities, reference)
    return float(_skm.log_loss(y_true, probs, labels=list(range(probs.shape[1]))))


def brier_score(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Binary Brier score on positive-class probabilities."""
    y_score, y_true = binary_scores(scores, reference)
    return float(_skm.brier_score_loss(y_true, y_score))


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    reference: Sequence[int],
    n_bins: int = 15,
) -> float:
    """ECE, in-house: equal-width confidence bins over the argmax prediction.

    No canonical torch-free implementation exists; the binning variant is fixed
    here (equal-width, top-1 confidence, L1) and stated in the suite signature.
    """
    probs, y_true = probs_2d(probabilities, reference)
    confidence = probs.max(axis=1)
    predicted_cls = probs.argmax(axis=1)
    correct = (predicted_cls == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)

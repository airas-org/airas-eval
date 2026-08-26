"""Classification metrics as pure functions: (predictions, references) -> score.

Deliberately dependency-light and deterministic. Every metric here is standard;
outputs are verified in tests against scikit-learn as the parity oracle.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

Average = Literal["micro", "macro", "weighted"]


def _paired_labels(
    predicted: Sequence[int], reference: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(predicted)
    y_true = np.asarray(reference)
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("predicted and reference must be 1-dimensional")
    if len(y_pred) != len(y_true):
        raise ValueError(f"length mismatch: {len(y_pred)} vs {len(y_true)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on zero examples")
    return y_pred, y_true


def _scores_and_labels(
    scores: Sequence[float], reference: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    y_score = np.asarray(scores, dtype=float)
    y_true = np.asarray(reference)
    if y_score.ndim != 1 or y_true.ndim != 1:
        raise ValueError("scores and reference must be 1-dimensional")
    if len(y_score) != len(y_true):
        raise ValueError(f"length mismatch: {len(y_score)} vs {len(y_true)}")
    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1} or len(labels) != 2:
        raise ValueError("reference must contain both classes 0 and 1")
    return y_score, y_true


def accuracy(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = _paired_labels(predicted, reference)
    return float(np.mean(y_pred == y_true))


def error_rate(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return 1.0 - accuracy(predicted, reference)


def top_k_accuracy(
    probabilities: Sequence[Sequence[float]], reference: Sequence[int], k: int
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(reference)
    if probs.ndim != 2:
        raise ValueError(
            f"probabilities must be 2-dimensional, got shape {probs.shape}"
        )
    if len(probs) != len(y_true):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(y_true)}")
    if not 1 <= k <= probs.shape[1]:
        raise ValueError(f"k={k} out of range for {probs.shape[1]} classes")
    top_k = np.argsort(probs, axis=1)[:, -k:]
    hits = (top_k == y_true[:, None]).any(axis=1)
    return float(np.mean(hits))


def _per_class_counts(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(classes, tp, fp, fn) over the union of observed classes."""
    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    tp = np.array([np.sum((y_pred == c) & (y_true == c)) for c in classes])
    fp = np.array([np.sum((y_pred == c) & (y_true != c)) for c in classes])
    fn = np.array([np.sum((y_pred != c) & (y_true == c)) for c in classes])
    return classes, tp, fp, fn


def _prf(
    predicted: Sequence[int], reference: Sequence[int], average: Average
) -> tuple[float, float, float]:
    y_pred, y_true = _paired_labels(predicted, reference)
    classes, tp, fp, fn = _per_class_counts(y_pred, y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_c = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall_c = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1_c = np.where(
            precision_c + recall_c > 0,
            2 * precision_c * recall_c / (precision_c + recall_c),
            0.0,
        )
    if average == "micro":
        tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
        p = tp_s / (tp_s + fp_s) if tp_s + fp_s > 0 else 0.0
        r = tp_s / (tp_s + fn_s) if tp_s + fn_s > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return float(p), float(r), float(f)
    if average == "macro":
        return float(precision_c.mean()), float(recall_c.mean()), float(f1_c.mean())
    support = np.array([np.sum(y_true == c) for c in classes], dtype=float)
    weights = support / support.sum()
    return (
        float(np.sum(precision_c * weights)),
        float(np.sum(recall_c * weights)),
        float(np.sum(f1_c * weights)),
    )


def precision(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    return _prf(predicted, reference, average)[0]


def recall(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    return _prf(predicted, reference, average)[1]


def f1(
    predicted: Sequence[int], reference: Sequence[int], average: Average = "macro"
) -> float:
    return _prf(predicted, reference, average)[2]


def macro_f1(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return f1(predicted, reference, average="macro")


def balanced_accuracy(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """Mean per-class recall over the classes present in the reference."""
    y_pred, y_true = _paired_labels(predicted, reference)
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls))


def matthews_corrcoef(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """Multiclass MCC (Gorodkin's R_K), matching scikit-learn."""
    y_pred, y_true = _paired_labels(predicted, reference)
    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    index = {c: i for i, c in enumerate(classes.tolist())}
    n = len(classes)
    confusion = np.zeros((n, n), dtype=float)
    for p, t in zip(y_pred.tolist(), y_true.tolist(), strict=False):
        confusion[index[t], index[p]] += 1
    t_sum = confusion.sum(axis=1)
    p_sum = confusion.sum(axis=0)
    total = confusion.sum()
    correct = np.trace(confusion)
    cov_ytyp = correct * total - float(t_sum @ p_sum)
    cov_ypyp = total**2 - float(p_sum @ p_sum)
    cov_ytyt = total**2 - float(t_sum @ t_sum)
    if cov_ypyp == 0 or cov_ytyt == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp))


def cohen_kappa(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = _paired_labels(predicted, reference)
    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    n = float(len(y_true))
    observed = float(np.mean(y_pred == y_true))
    expected = sum(
        (np.sum(y_true == c) / n) * (np.sum(y_pred == c) / n) for c in classes
    )
    if expected == 1.0:
        raise ValueError("Cohen's kappa is undefined when expected agreement is 1")
    return float((observed - expected) / (1.0 - expected))


def auroc(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Binary ROC AUC via the rank statistic (ties handled by mid-ranks)."""
    y_score, y_true = _scores_and_labels(scores, reference)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    order = np.argsort(y_score, kind="stable")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.mean(np.arange(i + 1, j + 2)))
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Area under the precision-recall curve (step-wise, scikit-learn variant)."""
    y_score, y_true = _scores_and_labels(scores, reference)
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    tp_cum = np.cumsum(y_sorted)
    n_pos = int(y_true.sum())
    precision_at = tp_cum / np.arange(1, len(y_sorted) + 1)
    recall_at = tp_cum / n_pos
    # Evaluate only at distinct-threshold boundaries (last index of each tie group).
    boundary = np.append(scores_sorted[1:] != scores_sorted[:-1], True)
    ap = 0.0
    prev_recall = 0.0
    for i in np.flatnonzero(boundary):
        ap += float(precision_at[i]) * (float(recall_at[i]) - prev_recall)
        prev_recall = float(recall_at[i])
    return float(ap)


def log_loss(
    probabilities: Sequence[Sequence[float]],
    reference: Sequence[int],
    eps: float = 1e-15,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(reference)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2-dimensional")
    if len(probs) != len(y_true):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(y_true)}")
    clipped = np.clip(probs, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    picked = clipped[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(picked)))


def brier_score(scores: Sequence[float], reference: Sequence[int]) -> float:
    """Binary Brier score: mean squared error of the positive-class probability."""
    y_score, y_true = _scores_and_labels(scores, reference)
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    reference: Sequence[int],
    n_bins: int = 15,
) -> float:
    """ECE with equal-width confidence bins over the argmax prediction."""
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(reference)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2-dimensional")
    if len(probs) != len(y_true):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(y_true)}")
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)

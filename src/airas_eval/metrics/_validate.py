"""Shared input validation, applied before delegating to canonical scorers.

Validation is ours (consistent, fail-closed errors); computation is theirs
(scikit-learn / scipy / domain-canonical packages).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def paired_1d(
    predicted: Sequence[Any], reference: Sequence[Any], dtype: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(predicted, dtype=dtype)
    y_true = np.asarray(reference, dtype=dtype)
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("predicted and reference must be 1-dimensional")
    if len(y_pred) != len(y_true):
        raise ValueError(f"length mismatch: {len(y_pred)} vs {len(y_true)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on zero examples")
    return y_pred, y_true


def probs_2d(
    probabilities: Sequence[Sequence[float]], reference: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(reference)
    if probs.ndim != 2:
        raise ValueError(
            f"probabilities must be 2-dimensional, got shape {probs.shape}"
        )
    if len(probs) != len(y_true):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(y_true)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on zero examples")
    return probs, y_true


def binary_scores(
    scores: Sequence[float], reference: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    y_score, y_true = paired_1d(scores, reference)
    y_score = y_score.astype(float)
    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1} or len(labels) != 2:
        raise ValueError("reference must contain both classes 0 and 1")
    return y_score, y_true

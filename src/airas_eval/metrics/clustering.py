"""Clustering agreement metrics (label-permutation invariant), pure functions."""

from collections.abc import Sequence

import numpy as np


def _contingency(
    labels_a: Sequence[int], labels_b: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("label arrays must be 1-dimensional")
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("cannot compute a metric on zero examples")
    classes_a, ia = np.unique(a, return_inverse=True)
    classes_b, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table, table.sum(axis=1), table.sum(axis=0)


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1) / 2.0


def adjusted_rand_index(predicted: Sequence[int], reference: Sequence[int]) -> float:
    table, rows, cols = _contingency(predicted, reference)
    n = table.sum()
    sum_comb = float(_comb2(table.astype(float)).sum())
    sum_rows = float(_comb2(rows.astype(float)).sum())
    sum_cols = float(_comb2(cols.astype(float)).sum())
    total = float(_comb2(np.array([n], dtype=float))[0])
    expected = sum_rows * sum_cols / total
    max_index = (sum_rows + sum_cols) / 2.0
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


def _mutual_info(table: np.ndarray) -> float:
    n = table.sum()
    rows = table.sum(axis=1, keepdims=True)
    cols = table.sum(axis=0, keepdims=True)
    mask = table > 0
    p = table[mask] / n
    outer = (rows @ cols)[mask] / (n * n)
    return float(np.sum(p * np.log(p / outer)))


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def normalized_mutual_info(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """NMI with arithmetic-mean normalization (scikit-learn default)."""
    table, rows, cols = _contingency(predicted, reference)
    h_pred, h_true = _entropy(rows), _entropy(cols)
    if h_pred == 0.0 and h_true == 0.0:
        return 1.0
    denom = (h_pred + h_true) / 2.0
    if denom == 0.0:
        return 0.0
    return float(_mutual_info(table) / denom)


def adjusted_mutual_info(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """AMI with arithmetic-mean normalization (scikit-learn default).

    Chance-corrected: preferred over NMI when cluster counts are large.
    """
    from math import exp, lgamma, log

    table, rows, cols = _contingency(predicted, reference)
    n = int(table.sum())
    mi = _mutual_info(table)
    h_pred, h_true = _entropy(rows), _entropy(cols)
    if h_pred == 0.0 and h_true == 0.0:
        return 1.0
    # Expected MI under the permutation model (hypergeometric marginals).
    emi = 0.0
    for a in rows.tolist():
        for b in cols.tolist():
            nij_lo = max(1, a + b - n)
            nij_hi = min(a, b)
            for nij in range(nij_lo, nij_hi + 1):
                term1 = (nij / n) * log(n * nij / (a * b))
                log_term2 = (
                    lgamma(a + 1)
                    + lgamma(b + 1)
                    + lgamma(n - a + 1)
                    + lgamma(n - b + 1)
                    - lgamma(n + 1)
                    - lgamma(nij + 1)
                    - lgamma(a - nij + 1)
                    - lgamma(b - nij + 1)
                    - lgamma(n - a - b + nij + 1)
                )
                emi += term1 * exp(log_term2)
    denom = (h_pred + h_true) / 2.0 - emi
    if denom == 0.0:
        return 0.0
    return float((mi - emi) / denom)


def v_measure(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """Harmonic mean of homogeneity and completeness."""
    table, rows, cols = _contingency(predicted, reference)
    mi = _mutual_info(table)
    h_pred, h_true = _entropy(rows), _entropy(cols)
    homogeneity = 1.0 if h_true == 0 else mi / h_true
    completeness = 1.0 if h_pred == 0 else mi / h_pred
    if homogeneity + completeness == 0.0:
        return 0.0
    return float(2 * homogeneity * completeness / (homogeneity + completeness))

"""Clustering agreement metrics, delegated to scikit-learn.

All are label-permutation invariant. Normalization variants are pinned:
NMI/AMI use arithmetic-mean normalization (the scikit-learn default), stated
in the suite signature.
"""

from collections.abc import Sequence

from sklearn import metrics as _skm

from airas_eval.metrics._validate import paired_1d


def adjusted_rand_index(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.adjusted_rand_score(y_true, y_pred))


def normalized_mutual_info(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(
        _skm.normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    )


def adjusted_mutual_info(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(
        _skm.adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    )


def v_measure(predicted: Sequence[int], reference: Sequence[int]) -> float:
    y_pred, y_true = paired_1d(predicted, reference)
    return float(_skm.v_measure_score(y_true, y_pred))

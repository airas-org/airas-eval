"""Set-recovery metrics: how well a predicted set of items matches a
reference set, with no notion of classes, order or scores. Items are any
hashable values; duplicates within a set are ignored."""

from collections.abc import Hashable, Iterable

from airas_eval.exceptions import UndefinedMetric


def _sets(
    predicted: Iterable[Hashable], reference: Iterable[Hashable]
) -> tuple[set[Hashable], set[Hashable]]:
    pred, ref = set(predicted), set(reference)
    if not ref:
        raise UndefinedMetric("recall is undefined for an empty reference set")
    if not pred:
        raise UndefinedMetric("precision is undefined for an empty predicted set")
    return pred, ref


def precision(predicted: Iterable[Hashable], reference: Iterable[Hashable]) -> float:
    pred, ref = _sets(predicted, reference)
    return len(pred & ref) / len(pred)


def recall(predicted: Iterable[Hashable], reference: Iterable[Hashable]) -> float:
    pred, ref = _sets(predicted, reference)
    return len(pred & ref) / len(ref)


def f1(predicted: Iterable[Hashable], reference: Iterable[Hashable]) -> float:
    p, r = precision(predicted, reference), recall(predicted, reference)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

"""Exceptions that carry the skip discipline of the scoring layer.

The evaluator converts ONLY these exceptions into ``skipped`` entries. Anything
else (a plain ValueError from malformed inputs, a bug in a metric) propagates
and fails the evaluation — a scoring layer that silently converts its own
bugs into "undefined on this data" cannot be trusted.
"""


class UndefinedMetric(ValueError):
    """The metric is mathematically undefined on this particular data.

    Raised by metric implementations (e.g. MAPE with a zero reference,
    Pearson r on a constant input). Subclasses ValueError so direct callers
    of the metric functions can keep catching ValueError.
    """


class NotApplicable(Exception):
    """The metric does not apply to this data shape or task variant.

    Raised by bundle adapters (e.g. AUROC on non-binary labels, top-5
    accuracy with 5 or fewer classes).
    """

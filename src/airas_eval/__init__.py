from importlib.metadata import version as _version

__version__ = _version("airas-eval")

from airas_eval.evaluator import (  # noqa: E402
    AggregateReport,
    ComparisonReport,
    EvaluationReport,
    aggregate_reports,
    compare,
    evaluate,
    validate_inputs,
)

__all__ = [
    "AggregateReport",
    "ComparisonReport",
    "EvaluationReport",
    "__version__",
    "aggregate_reports",
    "compare",
    "evaluate",
    "validate_inputs",
]

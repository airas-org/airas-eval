from importlib.metadata import version as _version

__version__ = _version("airas-eval")

from airas_eval.evaluator import EvaluationReport, evaluate  # noqa: E402

__all__ = ["EvaluationReport", "__version__", "evaluate"]

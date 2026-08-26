"""Typed input contracts, one pydantic model per bundle.

These models are the single source of input validation for both the Python
API and the ``airas-eval score`` JSON boundary (one group each). Everything is
JSON-representable by construction: numpy arrays are coerced to lists,
unknown keys are rejected, and non-finite values are rejected wherever a
finite number is expected.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class GroupInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _numpy_to_python(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value


def _check_finite(values: Any, name: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers")


class ClassificationInputs(GroupInputs):
    """Labels are integer-coded classes; probabilities are (n, n_classes)."""

    predicted_labels: list[int]
    reference_labels: list[int]
    probabilities: list[list[float]] | None = None

    @field_validator("probabilities")
    @classmethod
    def _finite(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is not None:
            _check_finite(value, "probabilities")
        return value


class SearchInputs(GroupInputs):
    """Scores of evaluated candidates in evaluation order (higher=better).

    ``oracle_best`` is the published optimum of the benchmark being searched,
    in the same unit as the scores. It is reference data: fixed by the
    experimental design and verified upstream, never chosen by the agent
    (a self-chosen oracle makes regret meaningless).
    """

    evaluated_scores: list[float]
    oracle_best: float | None = None

    @field_validator("evaluated_scores", "oracle_best")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "scores")
        return value


class CandidateRankingInputs(GroupInputs):
    """Predicted vs true scores over a fixed candidate set (higher=better)."""

    predicted_scores: list[float]
    reference_scores: list[float]

    @field_validator("predicted_scores", "reference_scores")
    @classmethod
    def _finite(cls, value: list[float]) -> list[float]:
        _check_finite(value, "scores")
        return value


class MultiobjectiveInputs(GroupInputs):
    """Objective vectors per candidate, lower-is-better in every objective
    (use error rate, not accuracy, next to parameter count / MACs)."""

    points: list[list[float]]
    reference_point: list[float] | None = None
    reference_front: list[list[float]] | None = None

    @field_validator("points", "reference_point", "reference_front")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "objective values")
        return value

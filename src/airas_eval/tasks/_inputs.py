"""Typed input contracts, one pydantic model per bundle.

These models are the single source of input validation for both the Python
API and the ``airas-eval score`` JSON boundary (one group each). Everything is
JSON-representable by construction: numpy arrays are coerced to lists,
unknown keys are rejected, and non-finite values are rejected wherever a
finite number is expected.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    predicted_labels: list[int] = Field(
        description="各事例の予測クラス(0 始まりの整数)。reference_labels と同じ長さ・順序。"
    )
    reference_labels: list[int] = Field(
        description="各事例の正解クラス(0 始まりの整数)。実験設計で固定された参照データ。"
    )
    probabilities: list[list[float]] | None = Field(
        default=None,
        description="各事例のクラス確率 (n_examples, n_classes)。省略すると確率系の指標"
        "(log_loss, ECE, top-5, AUROC 等)は skipped になる。",
    )

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

    evaluated_scores: list[float] = Field(
        description="探索が評価した候補のスコアを評価順に並べたもの。高いほど良い。"
    )
    oracle_best: float | None = Field(
        default=None,
        description="ベンチマークの公表最適値(evaluated_scores と同じ単位)。実験設計で固定し、"
        "agent は選ばない。省略すると regret 系は skipped。",
    )

    @field_validator("evaluated_scores", "oracle_best")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "scores")
        return value


class CandidateRankingInputs(GroupInputs):
    """Predicted vs true scores over a fixed candidate set (higher=better)."""

    predicted_scores: list[float] = Field(
        description="予測器・プロキシが各候補に与えたスコア。高いほど良い候補と予測。"
    )
    reference_scores: list[float] = Field(
        description="各候補の真のスコア(ベンチマークの学習済み精度など)。同じ長さ・順序。"
    )

    @field_validator("predicted_scores", "reference_scores")
    @classmethod
    def _finite(cls, value: list[float]) -> list[float]:
        _check_finite(value, "scores")
        return value


class MultiobjectiveInputs(GroupInputs):
    """Objective vectors per candidate, lower-is-better in every objective
    (use error rate, not accuracy, next to parameter count / MACs)."""

    points: list[list[float]] = Field(
        description="候補ごとの目的ベクトル (n_points, n_objectives)。全目的を最小化"
        "(精度ではなく誤り率、パラメータ数、MACs など)。"
    )
    reference_point: list[float] | None = Field(
        default=None,
        description="hypervolume の参照点(各目的の許容最悪値)。実験設計で固定し、結果を見て"
        "から選ばない。省略すると hypervolume は skipped。",
    )
    reference_front: list[list[float]] | None = Field(
        default=None,
        description="既知の参照 Pareto フロント。IGD/GD に使う。省略するとそれらは skipped。",
    )

    @field_validator("points", "reference_point", "reference_front")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "objective values")
        return value

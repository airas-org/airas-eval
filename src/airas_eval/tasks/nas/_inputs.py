"""NAS input contracts: one flat model per task.

Each task takes ONE inputs file. Fields a metric needs but that were not
provided make that metric ``skipped`` (with a code); the model only enforces
what must be there for the task to make sense at all.
"""

from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from airas_eval.tasks.generic._inputs import (
    ClassificationInputs,
    TaskInputs,
    _check_finite,
)


def _fractions(value: list[float] | None, name: str) -> list[float] | None:
    if value is not None:
        _check_finite(value, name)
        if any(not 0.0 <= v <= 1.0 for v in value):
            raise ValueError(f"{name} must lie in [0, 1]")
    return value


class NasPreTrainingInputs(TaskInputs):
    """学習前のアーキテクチャ性能: 探索軌跡(ベンチマーク参照/プロキシのスコア)と、
    予測器・ゼロコストプロキシの推定スコア。どちらか一方は必須。"""

    # The cross-field rule enforced by ``_consistent`` below, stated in the
    # JSON Schema too so ``airas-eval schema`` shows it.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "anyOf": [
                {"required": ["evaluated_scores"]},
                {"required": ["predicted_scores", "reference_scores"]},
            ],
            "dependentRequired": {
                "predicted_scores": ["reference_scores"],
                "reference_scores": ["predicted_scores"],
                "evaluation_costs": ["evaluated_scores"],
            },
        },
    )

    evaluated_scores: list[float] | None = Field(
        default=None,
        description="探索が評価した候補のスコアを評価順に並べたもの(ベンチマーク参照値や"
        "プロキシ値)。高いほど良い。探索ランがある研究では必須。",
    )
    oracle_best: float | None = Field(
        default=None,
        description="ベンチマークの公表最適値(evaluated_scores と同じ単位)。実験設計で固定し、"
        "agent は選ばない。省略すると regret 系は skipped。",
    )
    evaluation_costs: list[float] | None = Field(
        default=None,
        description="各評価候補の学習コスト(秒やエポックなど、ベンチマークの公表値)。"
        "evaluated_scores と同じ順序・長さ。省略すると wall-clock 軸の指標は skipped。",
    )
    search_space_scores: list[float] | None = Field(
        default=None,
        description="探索空間の全候補のスコア(表形式ベンチマークが公表)。探索空間内順位と"
        "ランダム探索ベースラインに使う。省略するとそれらは skipped。",
    )
    predicted_scores: list[float] | None = Field(
        default=None,
        description="予測器・プロキシが各候補に与えたスコア。高いほど良い候補と予測。"
        "reference_scores と対で与える。省略すると予測器の指標は skipped。",
    )
    reference_scores: list[float] | None = Field(
        default=None,
        description="predicted_scores の各候補の真のスコア(ベンチマークの学習済み精度など)。"
        "同じ長さ・順序。",
    )

    @field_validator(
        "evaluated_scores",
        "oracle_best",
        "evaluation_costs",
        "search_space_scores",
        "predicted_scores",
        "reference_scores",
    )
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "scores")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "NasPreTrainingInputs":
        has_search = self.evaluated_scores is not None
        has_predictor = (
            self.predicted_scores is not None or self.reference_scores is not None
        )
        if not has_search and not has_predictor:
            raise ValueError(
                "provide evaluated_scores (a search run) and/or predicted_scores + "
                "reference_scores (a predictor); neither was given"
            )
        if (self.predicted_scores is None) != (self.reference_scores is None):
            raise ValueError(
                "predicted_scores and reference_scores must be given together"
            )
        if self.evaluation_costs is not None:
            if self.evaluated_scores is None:
                raise ValueError("evaluation_costs given without evaluated_scores")
            if len(self.evaluation_costs) != len(self.evaluated_scores):
                raise ValueError(
                    "evaluation_costs must have one entry per evaluated score"
                )
        return self


class NasPostTrainingInputs(ClassificationInputs):
    """学習後のアーキテクチャ性能: 学習済み最終アーキテクチャの予測(必須)と、
    ベースライン・テスト最適値・精度–効率トレードオフの点集合(任意)。"""

    random_architecture_accuracies: list[float] | None = Field(
        default=None,
        description="同じ探索空間から一様に抽出し、同じパイプラインで学習したランダム"
        "アーキテクチャの正解率(0〜1)。Yang et al. 2020 のベースライン。",
    )
    oracle_test_best: float | None = Field(
        default=None,
        description="ベンチマークの公表テスト最適値(0〜1 の割合)。実験設計で固定し、agent は"
        "選ばない。省略すると test_regret は skipped。",
    )
    points: list[list[float]] | None = Field(
        default=None,
        description="精度–効率トレードオフの点集合 (n_points, n_objectives)。全目的を最小化"
        "(誤り率、パラメータ数、MACs など)。省略するとフロント系の指標は skipped。",
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

    @field_validator("random_architecture_accuracies")
    @classmethod
    def _finite_accs(cls, value: list[float] | None) -> list[float] | None:
        return _fractions(value, "random_architecture_accuracies")

    @field_validator("oracle_test_best")
    @classmethod
    def _finite_oracle(cls, value: float | None) -> float | None:
        if value is not None:
            _fractions([value], "oracle_test_best")
        return value

    @field_validator("points", "reference_point", "reference_front")
    @classmethod
    def _finite_objectives(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "objective values")
        return value

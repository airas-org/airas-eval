"""NAS-specific input contracts: the core models plus the reference data that
NAS benchmarks and evaluation protocols make available."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from airas_eval.tasks._inputs import ClassificationInputs, SearchInputs, _check_finite


class NasSearchInputs(SearchInputs):
    """A search run on a NAS benchmark (architecture performance looked up or
    estimated, not trained by the run).

    * ``evaluation_costs``: training cost of each evaluated architecture (same
      order as ``evaluated_scores``; seconds or epochs as the benchmark
      reports them). Gives the estimated-wall-clock axis of NAS-Bench-101/201.
    * ``search_space_scores``: the score of every architecture in the
      benchmark's search space (tabular benchmarks publish this). Enables the
      space-relative metrics and the random-search baseline.
    """

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

    @field_validator("evaluation_costs", "search_space_scores")
    @classmethod
    def _finite_extra(cls, value: Any) -> Any:
        if value is not None:
            _check_finite(value, "NAS reference data")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "NasSearchInputs":
        if self.evaluation_costs is not None and len(self.evaluation_costs) != len(
            self.evaluated_scores
        ):
            raise ValueError("evaluation_costs must have one entry per evaluated score")
        return self


def _fractions(value: list[float] | None, name: str) -> list[float] | None:
    if value is not None:
        _check_finite(value, name)
        if any(not 0.0 <= v <= 1.0 for v in value):
            raise ValueError(f"{name} must lie in [0, 1]")
    return value


class NasArchitectureInputs(ClassificationInputs):
    """A trained final architecture's predictions, plus (optionally):

    * ``random_architecture_accuracies``: accuracies of randomly sampled
      architectures trained with the same pipeline — the baseline Yang et al.
      (ICLR 2020) ask for;
    * ``oracle_test_best``: the benchmark's published test optimum as a
      fraction in [0, 1], for test regret. Reference data fixed by the
      experimental design, never chosen by the agent.
    """

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

    @field_validator("random_architecture_accuracies")
    @classmethod
    def _finite_accs(cls, value: list[float] | None) -> list[float] | None:
        return _fractions(value, "random_architecture_accuracies")

    @field_validator("oracle_test_best")
    @classmethod
    def _finite_oracle(cls, value: float | None) -> float | None:
        if value is None:
            return None
        _fractions([value], "oracle_test_best")
        return value

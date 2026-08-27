"""NAS-specific input contracts: the core models plus the reference data that
NAS benchmarks and evaluation protocols make available."""

from typing import Any

from pydantic import field_validator, model_validator

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

    evaluation_costs: list[float] | None = None
    search_space_scores: list[float] | None = None

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

    random_architecture_accuracies: list[float] | None = None
    oracle_test_best: float | None = None

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

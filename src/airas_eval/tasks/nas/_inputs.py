"""NAS-specific input contracts: the core models plus the reference data that
NAS benchmarks and evaluation protocols make available."""

from typing import Any

from pydantic import field_validator, model_validator

from airas_eval.tasks._inputs import ClassificationInputs, SearchInputs, _check_finite


class NasSearchInputs(SearchInputs):
    """A search run on a NAS benchmark.

    * ``evaluation_costs``: training cost of each evaluated architecture (same
      order as ``evaluated_scores``; seconds or epochs as the benchmark
      reports them). Gives the estimated-wall-clock axis of NAS-Bench-101/201.
    * ``search_space_scores``: the score of every architecture in the
      benchmark's search space (tabular benchmarks publish this). Enables the
      space-relative metrics and the random-search baseline.
    * ``final_test_score`` / ``oracle_test_best``: test-set score of the
      architecture the run selected, and the benchmark's test optimum, for a
      test regret separate from the validation regret used during search.
    """

    evaluation_costs: list[float] | None = None
    search_space_scores: list[float] | None = None
    final_test_score: float | None = None
    oracle_test_best: float | None = None

    @field_validator(
        "evaluation_costs",
        "search_space_scores",
        "final_test_score",
        "oracle_test_best",
    )
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
        if (self.final_test_score is None) != (self.oracle_test_best is None):
            raise ValueError(
                "final_test_score and oracle_test_best must be given together"
            )
        return self


class NasArchitectureInputs(ClassificationInputs):
    """A trained final architecture's predictions, plus (optionally) the
    accuracies of randomly sampled architectures trained with the same
    pipeline — the baseline Yang et al. (ICLR 2020) ask for."""

    random_architecture_accuracies: list[float] | None = None

    @field_validator("random_architecture_accuracies")
    @classmethod
    def _finite_accs(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            _check_finite(value, "random_architecture_accuracies")
            if any(not 0.0 <= v <= 1.0 for v in value):
                raise ValueError("random_architecture_accuracies must lie in [0, 1]")
        return value

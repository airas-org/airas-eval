"""Generic tasks: one evaluation family each, no research-area assumptions."""

from airas_eval.tasks.generic import (
    binary_classification,
    candidate_ranking,
    classification,
    multiobjective,
    search,
)

TASKS = (
    classification.TASK,
    binary_classification.TASK,
    search.TASK,
    candidate_ranking.TASK,
    multiobjective.TASK,
)

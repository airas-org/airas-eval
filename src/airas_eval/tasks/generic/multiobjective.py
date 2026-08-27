"""Generic task: the full multiobjective metric set over its own inputs."""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import MultiobjectiveInputs

TASK = TaskSpec.from_sets(
    "multiobjective", MultiobjectiveInputs, _metric_sets.MULTIOBJECTIVE
)

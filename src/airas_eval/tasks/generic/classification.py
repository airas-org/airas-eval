"""Generic task: the full classification metric set over its own inputs."""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import ClassificationInputs

TASK = TaskSpec.from_sets(
    "classification", ClassificationInputs, _metric_sets.CLASSIFICATION
)

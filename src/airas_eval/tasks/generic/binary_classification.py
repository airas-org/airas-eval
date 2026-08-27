"""Generic task: the full binary_classification metric set over its own inputs."""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import ClassificationInputs

TASK = TaskSpec.from_sets(
    "binary_classification", ClassificationInputs, _metric_sets.BINARY_CLASSIFICATION
)

"""Generic task: the full search metric set over its own inputs."""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import SearchInputs

TASK = TaskSpec.from_sets("search", SearchInputs, _metric_sets.SEARCH)

"""Generic task: the full classification bundle as a single required group."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="classification",
    groups=(Group("main", _bundles.CLASSIFICATION),),
)

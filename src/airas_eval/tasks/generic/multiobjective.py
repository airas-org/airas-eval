"""Generic task: the full multiobjective bundle as a single required group."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="multiobjective",
    groups=(Group("multiobjective", _bundles.MULTIOBJECTIVE),),
)

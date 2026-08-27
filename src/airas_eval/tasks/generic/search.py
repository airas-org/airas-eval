"""Generic task: the full search bundle as a single required group."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="search",
    groups=(Group("search", _bundles.SEARCH),),
)

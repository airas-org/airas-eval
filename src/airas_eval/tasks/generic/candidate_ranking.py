"""Generic task: the full candidate ranking bundle as a single required group."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="candidate_ranking",
    groups=(Group("candidate_ranking", _bundles.CANDIDATE_RANKING),),
)

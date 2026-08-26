"""NAS: an accuracy-vs-cost trade-off claim.

One (error, cost) vector per candidate, both minimized; the hypervolume
reference point is fixed in the experimental design."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="nas_tradeoff",
    description=__doc__ or "",
    groups=(Group("main", _bundles.MULTIOBJECTIVE),),
)

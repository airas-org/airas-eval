"""NAS on a tabular benchmark: search efficiency of one run.

Everything ``search`` reports, plus the NAS protocol: the incumbent against
estimated wall-clock cost (NAS-Bench-101/201), where the best found
architecture sits in the search space, what random search with the same
number of evaluations would be expected to reach, improvement over the
search-space mean (Yang et al. 2020), and test regret separate from the
validation regret used during search. Each NAS-specific block is optional
and its omission is reported."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks.nas import _bundles

TASK = TaskSpec(
    task_type="nas_search",
    description=__doc__ or "",
    groups=(Group("main", _bundles.SEARCH),),
)

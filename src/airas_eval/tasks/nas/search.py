"""NAS on a tabular benchmark: search efficiency of one run.

Inputs are the benchmark scores of the evaluated architectures in evaluation
order plus the benchmark's published optimum. The final architecture's
accuracy is a table lookup here, so there are no predictions to score — a
study that trains its found architecture evaluates that separately as
``nas_architecture``."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="nas_search",
    description=__doc__ or "",
    groups=(Group("main", _bundles.SEARCH),),
)

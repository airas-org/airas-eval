"""NAS: a performance predictor or zero-cost proxy claim.

Proxy scores versus benchmark ground truth over a fixed candidate pool."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="nas_predictor",
    description=__doc__ or "",
    groups=(Group("main", _bundles.CANDIDATE_RANKING),),
)

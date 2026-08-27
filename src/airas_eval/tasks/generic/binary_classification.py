"""Generic task: binary classification with positive-class scores."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="binary_classification",
    description=__doc__ or "",
    groups=(Group("binary_classification", _bundles.BINARY_CLASSIFICATION),),
)

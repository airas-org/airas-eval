"""NAS: target-task performance of a trained final architecture.

Predictions of the found architecture on the held-out test set. Single-label
multiclass (CIFAR / ImageNet style)."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles

TASK = TaskSpec(
    task_type="nas_architecture",
    description=__doc__ or "",
    groups=(Group("main", _bundles.CLASSIFICATION),),
)

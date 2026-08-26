"""NAS: target-task performance of a trained final architecture.

Everything ``classification`` reports, plus the baseline Yang et al. (ICLR
2020) ask for: accuracy relative to randomly sampled architectures trained
with the same pipeline, which factors out the search space and training
protocol from the claimed gain."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks.nas import _bundles

TASK = TaskSpec(
    task_type="nas_architecture",
    description=__doc__ or "",
    groups=(Group("main", _bundles.ARCHITECTURE),),
)

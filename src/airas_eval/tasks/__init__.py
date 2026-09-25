"""Task registry: the source of truth for task type -> metric mapping.

One sub-package per area: ``generic`` (evaluation families with no research
assumptions), one per research area (``nas``, ...) and one per fixed
benchmark (``scigym``, ...). Each sub-package has a
generated README listing exactly which metrics every task returns
(``python -m airas_eval.tasks.readme``), and ``airas-eval list`` prints the
same from the live registry. Only task types are registered — the metric
sets in ``_metric_sets`` are not.

The list is deliberately explicit (no entry points, no package scanning):
in a trust layer, what gets computed must be visible in a reviewed diff,
never injected by installing a package.
"""

from types import MappingProxyType

from airas_eval.spec import TaskSpec
from airas_eval.tasks import generic, nas, scigym

AREAS: dict[str, tuple[TaskSpec, ...]] = {
    "generic": generic.TASKS,
    "nas": nas.TASKS,
    "scigym": scigym.TASKS,
}


def _build() -> dict[str, TaskSpec]:
    tasks: dict[str, TaskSpec] = {}
    for area_tasks in AREAS.values():
        for task in area_tasks:
            if task.task_type in tasks:
                raise ValueError(f"duplicate task type {task.task_type!r}")
            tasks[task.task_type] = task
    return tasks


TASKS = MappingProxyType(_build())

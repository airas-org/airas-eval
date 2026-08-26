"""Neural architecture search: one task per kind of claim a NAS study makes.

Each task is the corresponding generic task plus what the NAS literature
adds (see ``_bundles``). Trade-off fronts have nothing NAS-specific: use the
generic ``multiobjective`` task.
"""

from airas_eval.tasks.nas import architecture, predictor, search

TASKS = (search.TASK, architecture.TASK, predictor.TASK)

"""Neural architecture search: one task per kind of claim a NAS study makes.

Each is a branch with its own evaluation data (a benchmark run, a trained
architecture, a predictor, a trade-off front), not a bag of everything NAS
might report — keeping each task small keeps the report legible.
"""

from airas_eval.tasks.nas import architecture, predictor, search, tradeoff

TASKS = (search.TASK, architecture.TASK, predictor.TASK, tradeoff.TASK)

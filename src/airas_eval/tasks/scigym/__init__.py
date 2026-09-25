"""SciGym (Duan et al. 2025): a fixed benchmark pack. Dataset pinned by
hash, the benchmark's own Evaluator pinned by commit, Table 1 as the
published reference."""

from airas_eval.tasks.scigym import small

TASKS = (small.TASK,)

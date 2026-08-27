"""Neural architecture search: two tasks, by when the architecture's
performance is measured.

* ``nas_pre_training`` — performance not trained by the run: benchmark lookup
  during search, predictors, zero-cost proxies.
* ``nas_post_training`` — performance of the trained final architecture.

Each is the corresponding generic task(s) plus what the NAS literature adds
(see ``_bundles``).
"""

from airas_eval.tasks.nas import post_training, pre_training

TASKS = (pre_training.TASK, post_training.TASK)

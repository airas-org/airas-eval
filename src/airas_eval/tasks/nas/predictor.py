"""NAS: a performance predictor or zero-cost proxy claim.

Everything ``candidate_ranking`` reports, plus Kendall / Spearman restricted
to the true top-10 % of the candidate pool — the NAS-Bench-Suite-Zero
protocol, because a proxy that ranks the whole space well but scrambles the
top is useless for search."""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks.nas import _bundles

TASK = TaskSpec(
    task_type="nas_predictor",
    description=__doc__ or "",
    groups=(Group("main", _bundles.PREDICTOR),),
)

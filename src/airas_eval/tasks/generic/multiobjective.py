"""多目的トレードオフの評価。各候補の目的ベクトル(全目的を最小化)から、Pareto
フロント、hypervolume、参照フロントとの距離を返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import MultiobjectiveInputs

TASK = TaskSpec.from_sets(
    "multiobjective",
    MultiobjectiveInputs,
    _metric_sets.MULTIOBJECTIVE,
    description=__doc__ or "",
)

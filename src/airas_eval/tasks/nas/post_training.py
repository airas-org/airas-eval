"""NAS の学習後評価。探索で選ばれ、実際に学習した最終アーキテクチャの性能を測る。

汎用 ``classification`` の全指標に、同じパイプラインで学習したランダムアーキテク
チャ群を基準にした相対精度(Yang et al., ICLR 2020)と、ベンチマークの公表テスト
最適値に対するテストリグレットを加えたもの。(誤り率, コスト) の点集合を与えれば、
汎用 ``multiobjective`` のフロント指標(hypervolume, IGD, ...)も合わせて返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets as core
from airas_eval.tasks.nas import _metric_sets
from airas_eval.tasks.nas._inputs import NasPostTrainingInputs

TASK = TaskSpec.from_sets(
    "nas_post_training",
    NasPostTrainingInputs,
    _metric_sets.ARCHITECTURE,
    core.MULTIOBJECTIVE,
    description=__doc__ or "",
)

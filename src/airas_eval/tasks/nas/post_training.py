"""NAS・学習後: 探索が選び、学習した最終アーキテクチャの性能。

汎用 ``classification`` の全指標に、Yang et al. (ICLR 2020) が求めるベースライン
(同じパイプラインで学習したランダムアーキテクチャ群に対する相対精度)と、
ベンチマークの公表テスト最適値に対するテストリグレットを加え、さらに
(誤り率, コスト) 点集合が与えられれば汎用 ``multiobjective`` のフロント指標
(hypervolume, IGD, ...)も返す。"""

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

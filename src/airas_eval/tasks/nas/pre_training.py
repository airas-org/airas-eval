"""NAS・学習前: この実行自身が学習していないアーキテクチャの性能。探索中に
表形式ベンチマークから参照したスコア、あるいは性能予測器・ゼロコストプロキシ
による推定値を評価する。

汎用 ``search`` タスクの全指標に NAS のプロトコル(推定 wall-clock コストに対する
暫定最良、探索空間内での位置、同じ予算のランダム探索ベースライン、探索空間平均に
対する改善)を加え、さらに汎用 ``candidate_ranking`` の全指標に真の上位 10% に限定
した Kendall / Spearman(NAS-Bench-Suite-Zero のプロトコル)を加えたもの。探索軌跡
か予測器スコアのどちらかは必須で、無い側の指標は理由付きで skipped になる。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.nas import _metric_sets
from airas_eval.tasks.nas._inputs import NasPreTrainingInputs

TASK = TaskSpec.from_sets(
    "nas_pre_training",
    NasPreTrainingInputs,
    _metric_sets.SEARCH,
    _metric_sets.PREDICTOR,
    description=__doc__ or "",
)

"""NAS・学習前: この実行自身が学習していないアーキテクチャの性能。探索中に
表形式ベンチマークから参照したスコア、あるいは性能予測器・ゼロコストプロキシ
による推定値を評価する。

任意グループ 2 つのうち少なくとも 1 つが必要。``search`` は汎用 ``search`` タスクの
全指標に NAS のプロトコルを加えたもの(推定 wall-clock コストに対する暫定最良、
探索空間内での位置、同じ予算のランダム探索ベースライン、探索空間平均に対する
改善)。``predictor`` は汎用 ``candidate_ranking`` の全指標に、真の上位 10% に限定した
Kendall / Spearman を加えたもの(NAS-Bench-Suite-Zero のプロトコル)。省略された
グループは黙って消えず、報告される。"""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks.nas import _bundles

TASK = TaskSpec(
    task_type="nas_pre_training",
    description=__doc__ or "",
    groups=(
        Group("search", _bundles.SEARCH, required=False),
        Group("predictor", _bundles.PREDICTOR, required=False),
    ),
)

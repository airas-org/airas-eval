"""NAS・学習後: 探索が選び、学習した最終アーキテクチャの性能。

必須グループ ``architecture`` は汎用 ``classification`` の全指標に、Yang et al.
(ICLR 2020) が求めるベースライン — 同じパイプラインで学習したランダムアーキテク
チャ群に対する相対精度 — と、ベンチマークの公表テスト最適値に対するテストリグ
レットを加えたもの。任意グループ ``tradeoff`` は (誤り率, コスト) 点集合に対する
汎用の多目的フロント評価で、精度と効率のトレードオフに関する主張のためのもの。"""

from airas_eval.spec import Group, TaskSpec
from airas_eval.tasks import _bundles as core
from airas_eval.tasks.nas import _bundles

TASK = TaskSpec(
    task_type="nas_post_training",
    description=__doc__ or "",
    groups=(
        Group("architecture", _bundles.ARCHITECTURE),
        Group("tradeoff", core.MULTIOBJECTIVE, required=False),
    ),
)

"""NAS の学習前評価。まだ学習していないアーキテクチャについて、探索の過程で得られた
性能の見積もりを評価する。対象は、表形式ベンチマークを参照して得た探索軌跡と、
性能予測器・ゼロコストプロキシが各候補に与えた推定スコアの 2 種類。

汎用 ``search`` タスクの全指標に NAS 固有の指標(推定 wall-clock コストに対する
暫定最良、探索空間内での位置、同じ予算のランダム探索との比較)を加え、汎用
``candidate_ranking`` の全指標に真の上位 10% に限定した順位相関
(NAS-Bench-Suite-Zero のプロトコル)を加えたもの。探索軌跡と予測器スコアは
どちらか一方があればよく、与えなかった側の指標は理由付きで skipped になる。"""

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

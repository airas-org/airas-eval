"""2 クラス分類の評価。ラベルは {0, 1}。クラス確率が与えられれば、AUROC・
平均適合率・Brier スコアなど閾値に依存しない指標も合わせて返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import ClassificationInputs

TASK = TaskSpec.from_sets(
    "binary_classification",
    ClassificationInputs,
    _metric_sets.BINARY_CLASSIFICATION,
    description=__doc__ or "",
)

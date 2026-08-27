"""多クラス分類の評価。予測ラベルと正解ラベル(任意でクラス確率)から、正解率・
適合率・再現率・F1・較正など、標準的な分類指標を一式返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import ClassificationInputs

TASK = TaskSpec.from_sets(
    "classification",
    ClassificationInputs,
    _metric_sets.CLASSIFICATION,
    description=__doc__ or "",
)

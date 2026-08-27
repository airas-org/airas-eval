"""探索・最適化の結果の評価。候補を順に評価していく過程のスコア列から、最良値、
既知の最適値に対するリグレット、anytime 性能(良い候補をどれだけ早く見つけたか)を
返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import SearchInputs

TASK = TaskSpec.from_sets(
    "search", SearchInputs, _metric_sets.SEARCH, description=__doc__ or ""
)

"""反応ネットワーク推定。反応を取り除いた生化学モデル(SBML など)を与えられた
エージェントが、摂動実験の時系列から欠損反応を推定して提出したモデルを、
真のモデルと比べて採点する。
インスタンスごとに、追加した反応と欠損反応の集合の一致(適合率・再現率・F1、modifier
まで要求する厳格版も)と、同じ条件でシミュレートした時系列の誤差(bounded sMAPE)を計算し、
全インスタンスの単純平均を返す。SciGym(Duan et al. 2025)の Table 1 と同じ定義。
シミュレーションは評価層では行わないので、時系列は実験側が同じ条件で得たものを渡す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.sysbio import _metric_sets
from airas_eval.tasks.sysbio._inputs import ReactionNetworkInferenceInputs

TASK = TaskSpec.from_sets(
    "reaction_network_inference",
    ReactionNetworkInferenceInputs,
    _metric_sets.REACTION_NETWORK_INFERENCE,
    description=__doc__ or "",
)

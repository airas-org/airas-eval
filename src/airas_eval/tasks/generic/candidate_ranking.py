"""候補スコアラー(性能予測器やプロキシ)の評価。予測スコアと真のスコアの順位相関と、
上位の候補をどれだけ正しく選び出せるかを返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import CandidateRankingInputs

TASK = TaskSpec.from_sets(
    "candidate_ranking",
    CandidateRankingInputs,
    _metric_sets.CANDIDATE_RANKING,
    description=__doc__ or "",
)

"""Generic task: the full candidate_ranking metric set over its own inputs."""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import CandidateRankingInputs

TASK = TaskSpec.from_sets(
    "candidate_ranking", CandidateRankingInputs, _metric_sets.CANDIDATE_RANKING
)

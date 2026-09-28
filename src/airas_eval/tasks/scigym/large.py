"""SciGym-large ベンチマーク(Duan et al. 2025 の公式リリース large split、213 件)。
small より大きな系(真のモデルの SBML はおよそ 3.5 倍の長さ)で、課題と採点は scigym_small と
同じ: 反応をすべて取り除いた SBML を渡されたエージェントが摂動実験から反応を推定して提出し、
公式 Evaluator で STE / RMS / NTS を採点する。データ(213 件の truth / partial / sedml)は
公式リリースの sha256 で固定する。論文は small でしか評価していないので Table 1 との差は返さない。
導入(extra、aarch64 の uv 設定、Dockerfile)は scigym_small の説明と同じ。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.scigym import _metric_sets
from airas_eval.tasks.scigym._inputs import ScigymLargeInputs

TASK = TaskSpec.from_sets(
    "scigym_large",
    ScigymLargeInputs,
    _metric_sets.SCIGYM_LARGE,
    description=__doc__ or "",
)

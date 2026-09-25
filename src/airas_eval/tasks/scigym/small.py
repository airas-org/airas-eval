"""SciGym-small ベンチマーク(Duan et al. 2025)。反応をすべて取り除いた SBML モデルを
渡されたエージェントが、摂動実験の時系列から反応を推定して提出したモデルを、
SciGym 公式の Evaluator でそのまま採点する。指標は論文の 3 つ: STE(軌道の sMAPE)、
RMS(反応の一致の P/R/F1、modifier なしと modifier ありの厳格版)、NTS(種間エッジの
P/R/F1 とエッジ型別の F1)。データ(137 件の truth / partial / sedml)は公式リリースの
sha256 で固定し、Table 1 の最良の公表値との差も返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.scigym import _metric_sets
from airas_eval.tasks.scigym._inputs import ScigymSmallInputs

TASK = TaskSpec.from_sets(
    "scigym_small",
    ScigymSmallInputs,
    _metric_sets.SCIGYM_SMALL,
    description=__doc__ or "",
)

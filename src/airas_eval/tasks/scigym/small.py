"""SciGym-small ベンチマーク(Duan et al. 2025)。反応をすべて取り除いた SBML モデルを
渡されたエージェントが、摂動実験の時系列から反応を推定して提出したモデルを、
SciGym 公式の Evaluator でそのまま採点する。データ(137 件の truth / partial /
sedml)は公式リリースの sha256 で固定し、指標は論文 Table 1 と同じ定義。
最良の公表値との差も返す。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.scigym import _metric_sets
from airas_eval.tasks.scigym._inputs import ScigymSmallInputs

TASK = TaskSpec.from_sets(
    "scigym_small",
    ScigymSmallInputs,
    _metric_sets.SCIGYM_SMALL,
    description=__doc__ or "",
)

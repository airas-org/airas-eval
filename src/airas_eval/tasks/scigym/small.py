"""SciGym-small ベンチマーク(Duan et al. 2025)。反応をすべて取り除いた SBML モデルを
渡されたエージェントが、摂動実験の時系列から反応を推定して提出したモデルを、
SciGym 公式の Evaluator でそのまま採点する。指標は論文の 3 つ: STE(軌道の sMAPE)、
RMS(反応の一致の P/R/F1、modifier なしと modifier ありの厳格版)、NTS(種間エッジの
P/R/F1 とエッジ型別の F1)。データ(137 件の truth / partial / sedml)は公式リリースの
sha256 で固定し、Table 1 の最良の公表値との差も返す。

導入: 依存は extra で持つので、研究リポジトリの eval グループは
``eval = ["airas-eval[scigym]"]``(airas-eval は git のタグ指定)。実験コードが公式 Controller を
使うなら ``[tool.uv] default-groups = ["eval"]`` で同じ SciGym を使う。aarch64(RIKYU)では
wheel の無い固定版を uv の設定で差し替える: ``[tool.uv]`` に ``prerelease = "allow"``、
``override-dependencies = ["antimony==2.14.0", "libroadrunner==2.7.0"]``、
``exclude-dependencies = ["rrplugins", "phrasedml", "python-libnuml", "python-libcombine"]``。
Dockerfile では pygraphviz の wheel が要る共有ライブラリ(apt: libglib2.0-0 libx11-6 libxext6
libxrender1 libexpat1)を入れ、除外した libcombine の代わりに空の ``libcombine.py`` を
site-packages に置く(tellurium が import 時に参照するだけ)。uv 管理の Python で必要な
libpython の先読みはパックが行う。"""

from airas_eval.spec import TaskSpec
from airas_eval.tasks.scigym import _metric_sets
from airas_eval.tasks.scigym._inputs import ScigymSmallInputs

TASK = TaskSpec.from_sets(
    "scigym_small",
    ScigymSmallInputs,
    _metric_sets.SCIGYM_SMALL,
    description=__doc__ or "",
)

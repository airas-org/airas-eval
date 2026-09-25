"""SciGym-small scored by SciGym's own ``Evaluator`` — the benchmark's
official implementation (Duan et al. 2025, github.com/h4duan/SciGym), pinned
to one commit. The evaluator simulates the reference and submitted models
itself (libroadrunner), so ``scigym`` and its simulator stack must be
installed from that commit; otherwise every metric is skipped as a missing
dependency. Table 1 of the paper is carried here as the published reference,
so a report says at once how far a run is from the best published number.
"""

import ctypes
import hashlib
import json
import sysconfig
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from airas_eval.spec import MetricBinding, MetricSet

SCIGYM_COMMIT = "88a7b93609e35b6ecb4eb343d816d6ff09256c6a"
SCIGYM_SOURCE = f"git+https://github.com/h4duan/SciGym@{SCIGYM_COMMIT}"

# 論文 Table 1(SciGym-small、各モデル 1 回実行の 137 件平均)。行の順は論文どおり
PUBLISHED: dict[str, dict[str, float]] = {
    "Gemini-2.5-Flash": {
        "trajectory_smape": 0.4181,
        "reaction_f1_with_modifiers": 0.1217,
        "reaction_f1": 0.2005,
    },
    "GPT-4.1-mini": {
        "trajectory_smape": 0.6007,
        "reaction_f1_with_modifiers": 0.1320,
        "reaction_f1": 0.2322,
    },
    "Claude-3.5-Haiku": {
        "trajectory_smape": 0.6281,
        "reaction_f1_with_modifiers": 0.0530,
        "reaction_f1": 0.0987,
    },
    "Gemini-2.5-Pro": {
        "trajectory_smape": 0.3212,
        "reaction_f1_with_modifiers": 0.1817,
        "reaction_f1": 0.3383,
    },
    "GPT-4.1": {
        "trajectory_smape": 0.4611,
        "reaction_f1_with_modifiers": 0.1740,
        "reaction_f1": 0.3038,
    },
    "Claude-3.7-Sonnet": {
        "trajectory_smape": 0.3615,
        "reaction_f1_with_modifiers": 0.1688,
        "reaction_f1": 0.3047,
    },
}
BEST_PUBLISHED = {"trajectory_smape": 0.3212, "reaction_f1": 0.3383}  # Gemini-2.5-Pro

Instances = list[dict[str, Any]]
_FILES = (
    ("reference_sbml", "truth.xml"),
    ("incomplete_sbml", "partial.xml"),
    ("reference_sedml", "truth.sedml"),
)
_cache: dict[str, list[dict[str, Any]]] = {}


def _scigym() -> tuple[Any, Any, Any]:
    """Import the pinned SciGym; an ImportError becomes a MISSING_DEPENDENCY skip."""
    try:
        origin = json.loads(
            metadata.distribution("scigym").read_text("direct_url.json") or "{}"
        )
    except metadata.PackageNotFoundError as err:
        raise ImportError(f"scigym is not installed; install {SCIGYM_SOURCE}") from err
    commit = origin.get("vcs_info", {}).get("commit_id")
    if commit != SCIGYM_COMMIT:
        raise ImportError(
            f"scigym must be installed from {SCIGYM_SOURCE}, found commit {commit}"
        )
    # uv 管理の Python では libroadrunner が libpython を見つけられないので先読みする
    libpython = (
        Path(sysconfig.get_config_var("LIBDIR") or "")
        / f"libpython{sysconfig.get_python_version()}.so.1.0"
    )
    if libpython.exists():
        ctypes.CDLL(str(libpython), mode=ctypes.RTLD_GLOBAL)
    import pygraphviz  # noqa: F401 - Evaluator が反応グラフの構築で遅延 import する
    from scigym.data import SBML
    from scigym.data.question import Question
    from scigym.eval import Evaluator

    return SBML, Question, Evaluator


def official_scores(instances: Instances) -> list[dict[str, Any]]:
    """Run SciGym's Evaluator once per instance (cached per inputs), exactly as
    the benchmark's Controller does at the end of a run: a missing or invalid
    submission is scored as the incomplete model with success=False."""
    key = hashlib.sha256(json.dumps(instances, sort_keys=True).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    SBML, Question, Evaluator = _scigym()
    scores = []
    for instance in instances:
        with tempfile.TemporaryDirectory() as tmp:
            for field, name in _FILES:
                Path(tmp, name).write_text(instance[field])
            question = Question(
                sbml_directory_path=tmp, task_difficulty="fully_observable"
            )
            evaluator = Evaluator(
                true_sbml=question.get_original_sbml(),
                incomplete_sbml=question.get_partial_sbml(),
                incomplete_runnable_sbml=question.get_runnable_partial_sbml(),
            )
            try:
                if not instance.get("submitted_sbml"):
                    raise ValueError("no submission")
                result = evaluator(pred_sbml=SBML(instance["submitted_sbml"])).to_dict()
                result["success"] = True
            except Exception:  # noqa: BLE001 - 公式 Controller と同じく無効な提出は不完全モデルで採点
                result = evaluator(pred_sbml=evaluator.incomplete_sbml).to_dict()
                result["success"] = False
        scores.append(result)
    _cache[key] = scores
    return scores


def mean_score(instances: Instances, key: str) -> float:
    scores = official_scores(instances)
    return float(sum(s[key] for s in scores) / len(scores))


def gap_to_best_published(instances: Instances, key: str, metric: str) -> float:
    return mean_score(instances, key) - BEST_PUBLISHED[metric]


def trajectory_fit(instances: Instances) -> list[float]:
    return [1.0 - s["observe_smape"] for s in official_scores(instances)]


def n_instances(instances: Instances) -> float:
    return float(len(instances))


def n_valid_submissions(instances: Instances) -> float:
    return float(sum(s["success"] for s in official_scores(instances)))


_IN = ("instances",)


def _metric(name: str, key: str, description: str, direction: str) -> MetricBinding:
    return MetricBinding(
        name,
        mean_score,
        _IN,
        {"key": key},
        description=description,
        value_range="[0, 1]",
        direction=direction,
    )


SCIGYM_SMALL = MetricSet(
    provenance_packages=("scigym", "libroadrunner", "python-libsbml"),
    notes=(
        f"SciGym 公式 Evaluator(コミット {SCIGYM_COMMIT[:7]})による採点。反応の一致は種 ID の集合で判定し、"
        "追加反応か欠損反応が空なら 0 点。軌道誤差は |pred - true| / (|pred| + |true|) の全種・全時点平均。"
        "提出が無い・無効なインスタンスは不完全モデルの採点値。全指標はインスタンスの単純平均"
    ),
    metrics=(
        _metric(
            "trajectory_smape",
            "observe_smape",
            "軌道誤差(STE)。提出モデルと真のモデルの時系列の sMAPE をインスタンスで平均した値。",
            "lower",
        ),
        _metric(
            "reaction_precision",
            "rp_precision",
            "反応の適合率(インスタンス平均)。追加した反応のうち反応物と生成物の集合が一致したものの割合。",
            "higher",
        ),
        _metric(
            "reaction_recall",
            "rp_recall",
            "反応の再現率(インスタンス平均)。取り除かれていた反応のうち一致するものが提出された割合。",
            "higher",
        ),
        _metric(
            "reaction_f1",
            "rp_f1",
            "反応の F1(インスタンス平均)。適合率と再現率の調和平均。",
            "higher",
        ),
        _metric(
            "reaction_precision_with_modifiers",
            "rpm_precision",
            "反応の適合率(インスタンス平均、modifier の集合の一致も要求する厳格版)。",
            "higher",
        ),
        _metric(
            "reaction_recall_with_modifiers",
            "rpm_recall",
            "反応の再現率(インスタンス平均、厳格版)。",
            "higher",
        ),
        _metric(
            "reaction_f1_with_modifiers",
            "rpm_f1",
            "反応の F1(インスタンス平均、厳格版)。",
            "higher",
        ),
        MetricBinding(
            "trajectory_smape_vs_best_published",
            gap_to_best_published,
            _IN,
            {"key": "observe_smape", "metric": "trajectory_smape"},
            description=f"STE と論文 Table 1 の最良値(Gemini-2.5-Pro、{BEST_PUBLISHED['trajectory_smape']})との差。負なら公表値を上回る。",
            value_range="[-1, 1]",
            direction="lower",
        ),
        MetricBinding(
            "reaction_f1_vs_best_published",
            gap_to_best_published,
            _IN,
            {"key": "rp_f1", "metric": "reaction_f1"},
            description=f"反応 F1 と論文 Table 1 の最良値(Gemini-2.5-Pro、{BEST_PUBLISHED['reaction_f1']})との差。正なら公表値を上回る。",
            value_range="[-1, 1]",
            direction="higher",
        ),
    ),
    summary=(
        MetricBinding(
            "n_instances", n_instances, _IN, description="評価したインスタンス数。"
        ),
        MetricBinding(
            "n_valid_submissions",
            n_valid_submissions,
            _IN,
            description="有効な SBML が提出され、そのまま採点されたインスタンス数。",
        ),
    ),
    per_example=(
        MetricBinding(
            "trajectory_fit",
            trajectory_fit,
            _IN,
            description="インスタンスごとの 1 - 軌道誤差(高いほど良い)。2 システムのペア比較(compare)に使う。",
        ),
    ),
)

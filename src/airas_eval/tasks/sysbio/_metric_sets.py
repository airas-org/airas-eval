"""Reaction network inference metric set: the SciGym scoring protocol
(Duan et al. 2025, "Measuring Scientific Capabilities of Language Models with
a Systems Biology Dry Lab") written as fixed bindings.

Per instance, reaction recovery is precision / recall / F1 between the set of
added reactions and the set of removed reactions, a reaction being matched
when its reactant and product sets agree (strict variant: modifiers too);
an instance with no added or no removed reactions scores 0 on all three,
as in the reference implementation. Trajectory error is the bounded sMAPE
over every species and time point, with an instance whose species or
lengths do not line up with the reference scoring 1 (the maximum). Every
task metric is the plain mean over instances, which is how the benchmark
tables report them.
"""

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics import regression as _reg
from airas_eval.metrics import sets as _sets
from airas_eval.spec import MetricBinding, MetricSet
from airas_eval.tasks.sysbio._inputs import Trajectory

# 検証済み入力は dict で渡される(evaluator が model_dump した形)
Reaction = dict[str, list[str]]
Reactions = list[list[Reaction]]
Trajectories = list[Trajectory]


def _key(reaction: Reaction, with_modifiers: bool) -> tuple[frozenset[str], ...]:
    key = (frozenset(reaction["reactants"]), frozenset(reaction["products"]))
    return key + (frozenset(reaction["modifiers"]),) if with_modifiers else key


def _instance_scores(
    predicted: list[Reaction], reference: list[Reaction], with_modifiers: bool
) -> tuple[float, float, float]:
    pred = {_key(r, with_modifiers) for r in predicted}
    ref = {_key(r, with_modifiers) for r in reference}
    try:
        return _sets.precision(pred, ref), _sets.recall(pred, ref), _sets.f1(pred, ref)
    except UndefinedMetric:  # ベンチマークの規約: どちらかが空なら 0 点
        return 0.0, 0.0, 0.0


def _mean_over_instances(
    predicted_reactions: Reactions,
    reference_reactions: Reactions,
    with_modifiers: bool,
    index: int,
) -> float:
    scores = [
        _instance_scores(p, r, with_modifiers)[index]
        for p, r in zip(predicted_reactions, reference_reactions, strict=True)
    ]
    return float(sum(scores) / len(scores))


def reaction_precision(
    predicted_reactions: Reactions, reference_reactions: Reactions, with_modifiers: bool
) -> float:
    return _mean_over_instances(
        predicted_reactions, reference_reactions, with_modifiers, 0
    )


def reaction_recall(
    predicted_reactions: Reactions, reference_reactions: Reactions, with_modifiers: bool
) -> float:
    return _mean_over_instances(
        predicted_reactions, reference_reactions, with_modifiers, 1
    )


def reaction_f1(
    predicted_reactions: Reactions, reference_reactions: Reactions, with_modifiers: bool
) -> float:
    return _mean_over_instances(
        predicted_reactions, reference_reactions, with_modifiers, 2
    )


def _instance_smape(predicted: Trajectory, reference: Trajectory) -> float:
    # 参照実装と同じ整列: 両方の種 ID の和集合を sorted し、片方に無い種があれば形が
    # 合わなくなるので最大誤差 1 とする
    keys = sorted(set(predicted) | set(reference))
    pred = [predicted[k] for k in keys if k in predicted]
    ref = [reference[k] for k in keys if k in reference]
    if len(pred) != len(ref) or any(
        len(a) != len(b) for a, b in zip(pred, ref, strict=True)
    ):
        return 1.0
    return _reg.smape_bounded(pred, ref)


def per_instance_trajectory_fit(
    predicted_trajectories: Trajectories, reference_trajectories: Trajectories
) -> list[float]:
    return [
        1.0 - _instance_smape(p, r)
        for p, r in zip(predicted_trajectories, reference_trajectories, strict=True)
    ]


def trajectory_smape(
    predicted_trajectories: Trajectories, reference_trajectories: Trajectories
) -> float:
    fits = per_instance_trajectory_fit(predicted_trajectories, reference_trajectories)
    return float(1.0 - sum(fits) / len(fits))


def n_instances(reference_reactions: Reactions) -> float:
    return float(len(reference_reactions))


_REACTIONS = ("predicted_reactions", "reference_reactions")
_TRAJECTORIES = ("predicted_trajectories", "reference_trajectories")


def _recovery_bindings(with_modifiers: bool) -> tuple[MetricBinding, ...]:
    suffix = "_with_modifiers" if with_modifiers else ""
    strict = (
        "反応物・生成物に加えて modifier の集合も一致した"
        if with_modifiers
        else "反応物と生成物の集合が一致した"
    )
    kwargs = {"with_modifiers": with_modifiers}
    return (
        MetricBinding(
            f"reaction_precision{suffix}",
            reaction_precision,
            _REACTIONS,
            kwargs,
            description=f"反応の適合率(インスタンス平均)。提出モデルが追加した反応のうち、{strict}ものの割合。",
            value_range="[0, 1]",
            direction="higher",
        ),
        MetricBinding(
            f"reaction_recall{suffix}",
            reaction_recall,
            _REACTIONS,
            kwargs,
            description=f"反応の再現率(インスタンス平均)。取り除かれていた反応のうち、{strict}ものが提出された割合。",
            value_range="[0, 1]",
            direction="higher",
        ),
        MetricBinding(
            f"reaction_f1{suffix}",
            reaction_f1,
            _REACTIONS,
            kwargs,
            description="反応の F1(インスタンス平均)。インスタンスごとの適合率と再現率の調和平均を単純平均した値。",
            value_range="[0, 1]",
            direction="higher",
        ),
    )


REACTION_NETWORK_INFERENCE = MetricSet(
    provenance_packages=("numpy",),
    notes=(
        "SciGym(Duan et al. 2025)の採点規約。反応の一致は種 ID の集合で判定し、順序と化学量論は無視。"
        "追加反応か欠損反応が空のインスタンスは適合率・再現率・F1 とも 0。軌道誤差は "
        "|pred - true| / (|pred| + |true|) を全種・全時点で平均した [0, 1] の sMAPE で、両方 0 の点は 0、"
        "種や長さが合わないインスタンスは 1。全指標はインスタンスの単純平均"
    ),
    metrics=(
        MetricBinding(
            "trajectory_smape",
            trajectory_smape,
            _TRAJECTORIES,
            description="軌道誤差(STE)。提出モデルと真のモデルの時系列の bounded sMAPE をインスタンスで平均した値。",
            value_range="[0, 1]",
            direction="lower",
        ),
    )
    + _recovery_bindings(with_modifiers=False)
    + _recovery_bindings(with_modifiers=True),
    summary=(
        MetricBinding(
            "n_instances",
            n_instances,
            ("reference_reactions",),
            description="評価したインスタンス(生化学モデル)の数。",
        ),
    ),
    per_example=(
        MetricBinding(
            "trajectory_fit",
            per_instance_trajectory_fit,
            _TRAJECTORIES,
            description="インスタンスごとの 1 - 軌道誤差(高いほど良い)。2 システムのペア比較(compare)に使う。",
        ),
    ),
)

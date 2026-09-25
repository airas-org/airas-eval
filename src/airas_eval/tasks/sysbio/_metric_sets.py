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

from airas_eval.metrics import classification as _cls
from airas_eval.metrics import regression as _reg
from airas_eval.spec import MetricBinding, MetricSet
from airas_eval.tasks.sysbio._inputs import Trajectory

# 検証済み入力は dict で渡される(evaluator が model_dump した形)
Reaction = dict[str, list[str]]
Reactions = list[list[Reaction]]
Trajectories = list[Trajectory]


def _key(reaction: Reaction, with_modifiers: bool) -> tuple[frozenset[str], ...]:
    key = (frozenset(reaction["reactants"]), frozenset(reaction["products"]))
    return key + (frozenset(reaction["modifiers"]),) if with_modifiers else key


def reaction_score(
    predicted_reactions: Reactions,
    reference_reactions: Reactions,
    metric: str,
    with_modifiers: bool,
) -> float:
    """Mean over instances of precision / recall / f1 between the two reaction
    sets, scored as binary labels over their union by ``metrics.classification``
    (zero_division=0 gives the benchmark's empty-set convention for free)."""
    scores = []
    for predicted, reference in zip(
        predicted_reactions, reference_reactions, strict=True
    ):
        pred = {_key(r, with_modifiers) for r in predicted}
        ref = {_key(r, with_modifiers) for r in reference}
        universe = list(pred | ref)
        y_pred = [int(k in pred) for k in universe]
        y_ref = [int(k in ref) for k in universe]
        fn = getattr(_cls, metric)
        scores.append(fn(y_pred, y_ref, average="binary") if universe else 0.0)
    return float(sum(scores) / len(scores))


def _instance_smape(predicted: Trajectory, reference: Trajectory) -> float:
    # 参照実装と同じ整列: 種 ID の和集合で並べ、片方に無い種や長さ違いは最大誤差 1
    keys = sorted(set(predicted) | set(reference))
    pred = [predicted[k] for k in keys if k in predicted]
    ref = [reference[k] for k in keys if k in reference]
    if [len(s) for s in pred] != [len(s) for s in ref]:
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
    return 1.0 - sum(fits) / len(fits)


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
    descriptions = {
        "precision": f"反応の適合率(インスタンス平均)。提出モデルが追加した反応のうち、{strict}ものの割合。",
        "recall": f"反応の再現率(インスタンス平均)。取り除かれていた反応のうち、{strict}ものが提出された割合。",
        "f1": "反応の F1(インスタンス平均)。インスタンスごとの適合率と再現率の調和平均を単純平均した値。",
    }
    return tuple(
        MetricBinding(
            f"reaction_{metric}{suffix}",
            reaction_score,
            _REACTIONS,
            {"metric": metric, "with_modifiers": with_modifiers},
            description=description,
            value_range="[0, 1]",
            direction="higher",
        )
        for metric, description in descriptions.items()
    )


REACTION_NETWORK_INFERENCE = MetricSet(
    provenance_packages=("numpy", "scikit-learn"),
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

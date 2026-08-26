"""The agent-facing API: one call per task type, the full standard suite back.

Agents do not implement evaluation scripts and do not pick metrics. They call
``evaluate(task_type, inputs)`` and receive every standard metric for that task
type at pinned variants. There is no parameter for choosing metrics or
variants — reporting all of them is what removes the cherry-picking degree of
freedom. Metrics that cannot be computed on the given inputs are reported under
``skipped`` with a reason, never silently dropped.

Every report embeds provenance: the airas-eval version, the resolved versions
of the packages that actually computed the numbers, a SHA-256 of the inputs,
and the suite signature (task type + pinned variants).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata as _metadata
from typing import Any

import numpy as np

from airas_eval import __version__
from airas_eval.metrics import (
    classification as _cls,
)
from airas_eval.metrics import (
    clustering as _clu,
)
from airas_eval.metrics import (
    nlp as _nlp,
)
from airas_eval.metrics import (
    ranking as _rank,
)
from airas_eval.metrics import (
    regression as _reg,
)
from airas_eval.metrics import (
    structure as _struct,
)
from airas_eval.metrics import (
    vision as _vis,
)


class SkipMetric(Exception):
    """Raised inside a suite computation when a metric does not apply."""


@dataclass(frozen=True)
class SuiteMetric:
    name: str
    requires: tuple[str, ...]
    compute: Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class Suite:
    task_type: str
    signature: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    metrics: tuple[SuiteMetric, ...]
    provenance_packages: tuple[str, ...]


@dataclass
class EvaluationReport:
    task_type: str
    metrics: dict[str, float]
    skipped: dict[str, str]
    provenance: dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _inputs_sha256(inputs: dict[str, Any]) -> str:
    canonical = json.dumps(_to_jsonable(inputs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _versions(packages: tuple[str, ...]) -> dict[str, str]:
    resolved: dict[str, str] = {"airas-eval": __version__}
    for package in packages:
        try:
            resolved[package] = _metadata.version(package)
        except _metadata.PackageNotFoundError:
            resolved[package] = "not installed"
    return resolved


# ---------------------------------------------------------------------------
# Suite computations. Each takes the validated inputs dict; raises SkipMetric
# when the metric does not apply to this data.


def _binary_positive_scores(inputs: dict[str, Any]) -> tuple[Any, Any]:
    probs = np.asarray(inputs["probabilities"], dtype=float)
    reference = np.asarray(inputs["reference_labels"])
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise SkipMetric("binary-only metric: probabilities are not 2-class")
    if set(np.unique(reference).tolist()) != {0, 1}:
        raise SkipMetric("binary-only metric: reference is not {0, 1}")
    return probs[:, 1], reference


def _top5(inputs: dict[str, Any]) -> float:
    probs = np.asarray(inputs["probabilities"], dtype=float)
    if probs.ndim != 2 or probs.shape[1] <= 5:
        raise SkipMetric("top-5 requires more than 5 classes")
    return _cls.top_k_accuracy(probs.tolist(), inputs["reference_labels"], k=5)


def _mape_or_skip(inputs: dict[str, Any]) -> float:
    try:
        return _reg.mape(inputs["predicted_values"], inputs["reference_values"])
    except ValueError as err:
        raise SkipMetric(str(err)) from err


def _ndcg(inputs: dict[str, Any]) -> float:
    relevances = inputs["relevances"]
    lists = inputs["ranked_lists"]
    scores = [
        _rank.ndcg_at_k(lst, rel, k=10)
        for lst, rel in zip(lists, relevances, strict=True)
    ]
    return float(np.mean(scores))


def _wrapped(fn: Callable[..., float], *args: Any) -> float:
    try:
        return fn(*args)
    except ImportError as err:
        raise SkipMetric(f"requires an optional dependency: {err}") from err


def _prf(metric: str, average: str) -> Callable[[dict[str, Any]], float]:
    def compute(inputs: dict[str, Any]) -> float:
        fn = getattr(_cls, metric)
        return float(
            fn(inputs["predicted_labels"], inputs["reference_labels"], average=average)
        )

    return compute


def _paired_regression(name: str) -> Callable[[dict[str, Any]], float]:
    def compute(inputs: dict[str, Any]) -> float:
        fn = getattr(_reg, name)
        return float(fn(inputs["predicted_values"], inputs["reference_values"]))

    return compute


def _paired_clustering(name: str) -> Callable[[dict[str, Any]], float]:
    def compute(inputs: dict[str, Any]) -> float:
        fn = getattr(_clu, name)
        return float(fn(inputs["predicted_labels"], inputs["reference_labels"]))

    return compute


def _mean_at_k(
    per_query: Callable[..., float], k: int
) -> Callable[[dict[str, Any]], float]:
    def compute(inputs: dict[str, Any]) -> float:
        pairs = zip(inputs["ranked_lists"], inputs["relevant_sets"], strict=True)
        return float(np.mean([per_query(lst, set(rel), k) for lst, rel in pairs]))

    return compute


_CLASSIFICATION = Suite(
    task_type="classification",
    signature=(
        "classification/v1|prf:zero_division=0|ece:15-bins,equal-width,top1"
        "|binary-scores:probabilities[:,1]"
    ),
    required_inputs=("predicted_labels", "reference_labels"),
    optional_inputs=("probabilities",),
    provenance_packages=("numpy", "scikit-learn"),
    metrics=(
        SuiteMetric(
            "accuracy",
            ("predicted_labels", "reference_labels"),
            lambda x: _cls.accuracy(x["predicted_labels"], x["reference_labels"]),
        ),
        SuiteMetric(
            "error_rate",
            ("predicted_labels", "reference_labels"),
            lambda x: _cls.error_rate(x["predicted_labels"], x["reference_labels"]),
        ),
        *(
            SuiteMetric(
                f"{metric}_{avg}",
                ("predicted_labels", "reference_labels"),
                _prf(metric, avg),
            )
            for metric in ("precision", "recall", "f1")
            for avg in ("macro", "micro", "weighted")
        ),
        SuiteMetric(
            "balanced_accuracy",
            ("predicted_labels", "reference_labels"),
            lambda x: _cls.balanced_accuracy(
                x["predicted_labels"], x["reference_labels"]
            ),
        ),
        SuiteMetric(
            "matthews_corrcoef",
            ("predicted_labels", "reference_labels"),
            lambda x: _cls.matthews_corrcoef(
                x["predicted_labels"], x["reference_labels"]
            ),
        ),
        SuiteMetric(
            "cohen_kappa",
            ("predicted_labels", "reference_labels"),
            lambda x: _cls.cohen_kappa(x["predicted_labels"], x["reference_labels"]),
        ),
        SuiteMetric(
            "log_loss",
            ("probabilities", "reference_labels"),
            lambda x: _cls.log_loss(x["probabilities"], x["reference_labels"]),
        ),
        SuiteMetric(
            "expected_calibration_error",
            ("probabilities", "reference_labels"),
            lambda x: _cls.expected_calibration_error(
                x["probabilities"], x["reference_labels"]
            ),
        ),
        SuiteMetric("top_5_accuracy", ("probabilities", "reference_labels"), _top5),
        SuiteMetric(
            "auroc",
            ("probabilities", "reference_labels"),
            lambda x: _cls.auroc(*_binary_positive_scores(x)),
        ),
        SuiteMetric(
            "average_precision",
            ("probabilities", "reference_labels"),
            lambda x: _cls.average_precision(*_binary_positive_scores(x)),
        ),
        SuiteMetric(
            "brier_score",
            ("probabilities", "reference_labels"),
            lambda x: _cls.brier_score(*_binary_positive_scores(x)),
        ),
    ),
)

_REGRESSION = Suite(
    task_type="regression",
    signature="regression/v1|mape:strict-no-eps|smape:2|d|/(|p|+|t|)|kendall:tau-b",
    required_inputs=("predicted_values", "reference_values"),
    optional_inputs=(),
    provenance_packages=("numpy", "scikit-learn", "scipy"),
    metrics=(
        *(
            SuiteMetric(
                name,
                ("predicted_values", "reference_values"),
                _paired_regression(name),
            )
            for name in (
                "mse",
                "rmse",
                "mae",
                "smape",
                "r2_score",
                "explained_variance",
                "pearson_r",
                "spearman_rho",
                "kendall_tau",
            )
        ),
        SuiteMetric("mape", ("predicted_values", "reference_values"), _mape_or_skip),
    ),
)

_CLUSTERING = Suite(
    task_type="clustering",
    signature="clustering/v1|nmi,ami:arithmetic",
    required_inputs=("predicted_labels", "reference_labels"),
    optional_inputs=(),
    provenance_packages=("numpy", "scikit-learn"),
    metrics=tuple(
        SuiteMetric(
            name,
            ("predicted_labels", "reference_labels"),
            _paired_clustering(name),
        )
        for name in (
            "adjusted_rand_index",
            "normalized_mutual_info",
            "adjusted_mutual_info",
            "v_measure",
        )
    ),
)

_RETRIEVAL = Suite(
    task_type="retrieval",
    signature="retrieval/v1|k:1,5,10|map:full-list|ndcg:k=10,linear-gain",
    required_inputs=("ranked_lists", "relevant_sets"),
    optional_inputs=("relevances",),
    provenance_packages=("numpy",),
    metrics=(
        *(
            SuiteMetric(
                f"precision_at_{k}",
                ("ranked_lists", "relevant_sets"),
                _mean_at_k(_rank.precision_at_k, k),
            )
            for k in (1, 5, 10)
        ),
        *(
            SuiteMetric(
                f"recall_at_{k}",
                ("ranked_lists", "relevant_sets"),
                _mean_at_k(_rank.recall_at_k, k),
            )
            for k in (5, 10)
        ),
        SuiteMetric(
            "mean_reciprocal_rank",
            ("ranked_lists", "relevant_sets"),
            lambda x: _rank.mean_reciprocal_rank(
                x["ranked_lists"], [set(rel) for rel in x["relevant_sets"]]
            ),
        ),
        SuiteMetric(
            "mean_average_precision",
            ("ranked_lists", "relevant_sets"),
            lambda x: _rank.mean_average_precision(
                x["ranked_lists"], [set(rel) for rel in x["relevant_sets"]]
            ),
        ),
        SuiteMetric("ndcg_at_10", ("ranked_lists", "relevances"), _ndcg),
    ),
)

_TEXT_QA = Suite(
    task_type="text_qa",
    signature="text_qa/v1|normalization:squad-en",
    required_inputs=("predicted_texts", "reference_texts"),
    optional_inputs=(),
    provenance_packages=("numpy",),
    metrics=(
        SuiteMetric(
            "exact_match",
            ("predicted_texts", "reference_texts"),
            lambda x: _nlp.exact_match(x["predicted_texts"], x["reference_texts"]),
        ),
        SuiteMetric(
            "token_f1",
            ("predicted_texts", "reference_texts"),
            lambda x: _nlp.token_f1(x["predicted_texts"], x["reference_texts"]),
        ),
    ),
)

_TEXT_GENERATION = Suite(
    task_type="text_generation",
    signature="text_generation/v1|bleu,chrf:sacrebleu-defaults|rouge_l:stemmer",
    required_inputs=("predicted_texts", "reference_texts"),
    optional_inputs=(),
    provenance_packages=("numpy", "sacrebleu", "rouge-score"),
    metrics=(
        SuiteMetric(
            "bleu",
            ("predicted_texts", "reference_texts"),
            lambda x: _wrapped(
                _nlp.bleu,
                x["predicted_texts"],
                [[ref] for ref in x["reference_texts"]],
            ),
        ),
        SuiteMetric(
            "chrf",
            ("predicted_texts", "reference_texts"),
            lambda x: _wrapped(
                _nlp.chrf,
                x["predicted_texts"],
                [[ref] for ref in x["reference_texts"]],
            ),
        ),
        SuiteMetric(
            "rouge_l",
            ("predicted_texts", "reference_texts"),
            lambda x: _wrapped(
                _nlp.rouge_l, x["predicted_texts"], x["reference_texts"]
            ),
        ),
    ),
)

_SEGMENTATION = Suite(
    task_type="segmentation",
    signature="segmentation/v1|miou:aggregate-over-given-masks",
    required_inputs=("predicted_mask", "reference_mask"),
    optional_inputs=(),
    provenance_packages=("numpy",),
    metrics=(
        SuiteMetric(
            "pixel_accuracy",
            ("predicted_mask", "reference_mask"),
            lambda x: _vis.pixel_accuracy(x["predicted_mask"], x["reference_mask"]),
        ),
        SuiteMetric(
            "mean_iou",
            ("predicted_mask", "reference_mask"),
            lambda x: _vis.mean_iou(x["predicted_mask"], x["reference_mask"]),
        ),
        SuiteMetric(
            "dice_coefficient",
            ("predicted_mask", "reference_mask"),
            lambda x: _vis.dice_coefficient(x["predicted_mask"], x["reference_mask"]),
        ),
    ),
)

_STRUCTURE = Suite(
    task_type="structure_comparison",
    signature="structure_comparison/v1|rmsd:kabsch|tm:tmtools-norm-chain2",
    required_inputs=("predicted_coords", "reference_coords"),
    optional_inputs=("reference_sequence",),
    provenance_packages=("numpy", "tmtools"),
    metrics=(
        SuiteMetric(
            "rmsd",
            ("predicted_coords", "reference_coords"),
            lambda x: _struct.rmsd(
                x["predicted_coords"], x["reference_coords"], superpose=True
            ),
        ),
        SuiteMetric(
            "rmsd_no_superposition",
            ("predicted_coords", "reference_coords"),
            lambda x: _struct.rmsd(
                x["predicted_coords"], x["reference_coords"], superpose=False
            ),
        ),
        SuiteMetric(
            "tm_score",
            ("predicted_coords", "reference_coords", "reference_sequence"),
            lambda x: _wrapped(
                _struct.tm_score,
                x["predicted_coords"],
                x["reference_coords"],
                x["reference_sequence"],
            ),
        ),
    ),
)

SUITES: dict[str, Suite] = {
    suite.task_type: suite
    for suite in (
        _CLASSIFICATION,
        _REGRESSION,
        _CLUSTERING,
        _RETRIEVAL,
        _TEXT_QA,
        _TEXT_GENERATION,
        _SEGMENTATION,
        _STRUCTURE,
    )
}


def evaluate(task_type: str, inputs: dict[str, Any]) -> EvaluationReport:
    """Compute the full pinned metric suite for ``task_type`` on ``inputs``.

    Fail-closed on the contract: unknown task types, unknown input keys, and
    missing required inputs raise. Individual metrics that do not apply to the
    given data (wrong class count, missing optional input, missing optional
    dependency, mathematically undefined) are reported under ``skipped`` with a
    reason.
    """
    if task_type not in SUITES:
        raise KeyError(f"unknown task type {task_type!r}; known: {sorted(SUITES)}")
    suite = SUITES[task_type]

    allowed = set(suite.required_inputs) | set(suite.optional_inputs)
    unknown = set(inputs) - allowed
    if unknown:
        raise ValueError(
            f"unknown input keys {sorted(unknown)} for task {task_type!r}; "
            f"allowed: {sorted(allowed)}"
        )
    missing = [key for key in suite.required_inputs if key not in inputs]
    if missing:
        raise ValueError(f"missing required inputs for {task_type!r}: {missing}")

    metrics: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for metric in suite.metrics:
        absent = [key for key in metric.requires if key not in inputs]
        if absent:
            skipped[metric.name] = f"requires input(s): {', '.join(absent)}"
            continue
        try:
            metrics[metric.name] = float(metric.compute(inputs))
        except SkipMetric as reason:
            skipped[metric.name] = str(reason)
        except ValueError as err:
            skipped[metric.name] = f"undefined on this data: {err}"

    return EvaluationReport(
        task_type=task_type,
        metrics=metrics,
        skipped=skipped,
        provenance={
            "suite_signature": suite.signature,
            "versions": _versions(suite.provenance_packages),
            "inputs_sha256": _inputs_sha256(inputs),
        },
    )

"""The scoring entry points: ``evaluate``, ``aggregate_reports``, ``compare``.

``evaluate(task_type, inputs)`` computes the full fixed metric set for a task
type from ONE inputs dict — there is no parameter for choosing metrics,
variants, or a subset of the task. What cannot be computed is reported under
``skipped`` with a machine-readable code, never silently dropped; and ONLY
the dedicated skip exceptions are converted to skips — a plain ValueError
(malformed inputs, a bug in a metric) fails the evaluation.

Note on trust: this library cannot defend itself inside an agent-controlled
process (anything can be monkey-patched there). The trustworthy boundary is
the ``airas-eval score`` CLI run from a pinned environment the agent cannot
edit; the Python API is a convenience for that job and for humans.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as _metadata
from typing import Any

import numpy as np
from pydantic import ValidationError

from airas_eval import __version__
from airas_eval.exceptions import NotApplicable, UndefinedMetric
from airas_eval.metrics import stats as _stats
from airas_eval.spec import MetricBinding, SkipCode, TaskSpec


@dataclass
class EvaluationReport:
    task_type: str
    metrics: dict[str, float]
    curves: dict[str, list[Any]]
    skipped: dict[str, dict[str, str]]
    inputs_summary: dict[str, float]
    omitted_optional_inputs: list[str]
    provenance: dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True, allow_nan=False)


@dataclass
class AggregateReport:
    """Mean ± std over repeated runs (seeds) of the same task signature."""

    task_type: str
    n_reports: int
    metrics: dict[str, dict[str, float]]
    inputs_summary: dict[str, dict[str, float]]
    incomplete: dict[str, int]
    provenance: dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True, allow_nan=False)


@dataclass
class ComparisonReport:
    """Paired comparison of two systems on the same examples."""

    task_type: str
    comparisons: dict[str, dict[str, float]]
    provenance: dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True, allow_nan=False)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _inputs_sha256(payload: Any) -> str:
    canonical = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _versions(packages: tuple[str, ...]) -> dict[str, str]:
    resolved: dict[str, str] = {"airas-eval": __version__}
    for package in packages:
        try:
            resolved[package] = _metadata.version(package)
        except _metadata.PackageNotFoundError:
            resolved[package] = "not installed"
    return resolved


def _get_task(task_type: str) -> TaskSpec:
    from airas_eval.tasks import TASKS  # noqa: PLC0415 - avoids import cycle

    if task_type not in TASKS:
        raise KeyError(f"unknown task type {task_type!r}; known: {sorted(TASKS)}")
    return TASKS[task_type]


def _validated(task: TaskSpec, inputs: dict[str, Any]) -> dict[str, Any]:
    """Validate one inputs dict against the task contract; fail closed."""
    if not isinstance(inputs, dict):
        raise ValueError(
            f"invalid inputs for task {task.task_type!r}: expected an object with "
            f"fields {list(task.input_model.model_fields)}"
        )
    try:
        model = task.input_model.model_validate(inputs)
    except ValidationError as err:
        raise ValueError(f"invalid inputs for task {task.task_type!r}: {err}") from err
    return model.model_dump()


def validate_inputs(task_type: str, inputs: dict[str, Any]) -> list[str]:
    """Check an inputs dict against the task contract without scoring.

    Returns the optional inputs that were provided; raises ValueError with
    the same messages ``evaluate`` would.
    """
    task = _get_task(task_type)
    data = _validated(task, inputs)
    return [k for k in task.optional_inputs() if data.get(k) is not None]


def _run_binding(
    binding: MetricBinding, data: dict[str, Any]
) -> tuple[Any, dict[str, str] | None]:
    absent = [key for key in binding.inputs if data.get(key) is None]
    if absent:
        return None, {
            "code": SkipCode.MISSING_OPTIONAL_INPUT.value,
            "reason": f"requires input(s): {', '.join(absent)}",
        }
    try:
        value = binding.fn(*(data[key] for key in binding.inputs), **binding.kwargs)
    except NotApplicable as err:
        return None, {"code": SkipCode.NOT_APPLICABLE.value, "reason": str(err)}
    except UndefinedMetric as err:
        return None, {"code": SkipCode.UNDEFINED_ON_DATA.value, "reason": str(err)}
    except ImportError as err:
        return None, {
            "code": SkipCode.MISSING_DEPENDENCY.value,
            "reason": f"requires an optional dependency: {err}",
        }
    return value, None


def evaluate(task_type: str, inputs: dict[str, Any]) -> EvaluationReport:
    """Compute the full fixed metric set for ``task_type`` on ``inputs``.

    Fail-closed on the contract: unknown task types, unknown input keys,
    wrong types, and missing required inputs raise. Metrics that do not
    apply to the given data are reported under ``skipped`` with a
    machine-readable code and a reason; omitted optional inputs are listed.
    """
    task = _get_task(task_type)
    data = _validated(task, inputs)

    metrics: dict[str, float] = {}
    curves: dict[str, list[Any]] = {}
    skipped: dict[str, dict[str, str]] = {}
    inputs_summary: dict[str, float] = {}

    for target, bindings in ((metrics, task.metrics), (inputs_summary, task.summary)):
        for binding in bindings:
            value, skip = _run_binding(binding, data)
            if skip is not None:
                skipped[binding.name] = skip
                continue
            score = float(value)
            if not np.isfinite(score):
                skipped[binding.name] = {
                    "code": SkipCode.UNDEFINED_ON_DATA.value,
                    "reason": f"non-finite result ({score})",
                }
                continue
            target[binding.name] = score

    for binding in task.curves:
        value, skip = _run_binding(binding, data)
        if skip is not None:
            skipped[binding.name] = skip
            continue
        curve = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(curve)):
            skipped[binding.name] = {
                "code": SkipCode.UNDEFINED_ON_DATA.value,
                "reason": "non-finite values in curve",
            }
            continue
        curves[binding.name] = curve.tolist()

    return EvaluationReport(
        task_type=task_type,
        metrics=metrics,
        curves=curves,
        skipped=skipped,
        inputs_summary=inputs_summary,
        omitted_optional_inputs=[
            k for k in task.optional_inputs() if data.get(k) is None
        ],
        provenance={
            "task_signature": task.signature(),
            "versions": _versions(task.provenance_packages),
            "inputs_sha256": _inputs_sha256(data),
        },
    )


def aggregate_reports(reports: Sequence[dict[str, Any]]) -> AggregateReport:
    """Mean ± sample std per metric over repeated runs (e.g. seeds).

    Takes report dicts (``EvaluationReport`` as JSON). All reports must share
    one task signature — mixing versions or task types is a contract
    violation. A metric missing from some reports is listed under
    ``incomplete`` (with how many reports had it) instead of being averaged
    over a shifting subset.
    """
    if not reports:
        raise ValueError("need at least one report to aggregate")
    signatures = {r["provenance"]["task_signature"] for r in reports}
    if len(signatures) != 1:
        raise ValueError(
            f"cannot aggregate across task signatures: {sorted(signatures)}"
        )

    def _aggregate(key: str) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
        names = sorted({name for r in reports for name in r[key]})
        full: dict[str, dict[str, float]] = {}
        partial: dict[str, int] = {}
        for name in names:
            values = [r[key][name] for r in reports if name in r[key]]
            if len(values) != len(reports):
                partial[name] = len(values)
                continue
            full[name] = _stats.mean_std(values)
        return full, partial

    metrics, incomplete = _aggregate("metrics")
    summary, incomplete_summary = _aggregate("inputs_summary")
    incomplete.update(incomplete_summary)
    first = reports[0]
    return AggregateReport(
        task_type=first["task_type"],
        n_reports=len(reports),
        metrics=metrics,
        inputs_summary=summary,
        incomplete=incomplete,
        provenance={
            "task_signature": first["provenance"]["task_signature"],
            "versions": first["provenance"]["versions"],
            "inputs_sha256": [r["provenance"]["inputs_sha256"] for r in reports],
        },
    )


def compare(
    task_type: str, inputs_a: dict[str, Any], inputs_b: dict[str, Any]
) -> ComparisonReport:
    """Paired significance test between two systems on the same examples.

    Uses the task's per-example scores (first bound input = the prediction,
    the rest = reference data that must be identical on both sides) and a
    two-sided sign-flip permutation test. Provided so research agents never
    have to implement their own significance testing.
    """
    task = _get_task(task_type)
    if not task.per_example:
        raise ValueError(
            f"task type {task_type!r} declares no per-example score; paired "
            "comparison is not available for it"
        )
    data_a = _validated(task, inputs_a)
    data_b = _validated(task, inputs_b)
    comparisons: dict[str, dict[str, float]] = {}

    for binding in task.per_example:
        reference_keys = binding.inputs[1:]
        if any(
            _inputs_sha256(data_a.get(k)) != _inputs_sha256(data_b.get(k))
            for k in reference_keys
        ):
            raise ValueError(
                f"{binding.name}: paired comparison requires identical reference "
                f"inputs on both sides ({', '.join(reference_keys)})"
            )
        a = np.asarray(
            binding.fn(*(data_a[k] for k in binding.inputs), **binding.kwargs),
            dtype=float,
        )
        b = np.asarray(
            binding.fn(*(data_b[k] for k in binding.inputs), **binding.kwargs),
            dtype=float,
        )
        if a.shape != b.shape:
            raise ValueError(
                f"{binding.name}: per-example score shapes differ: "
                f"{a.shape} vs {b.shape}"
            )
        comparisons[binding.name] = {
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff": float((a - b).mean()),
            "p_value": _stats.paired_permutation_test(a.tolist(), b.tolist()),
            "n_examples": float(len(a)),
        }

    return ComparisonReport(
        task_type=task_type,
        comparisons=comparisons,
        provenance={
            "task_signature": task.signature(),
            "versions": _versions(task.provenance_packages),
            "inputs_a_sha256": _inputs_sha256(data_a),
            "inputs_b_sha256": _inputs_sha256(data_b),
            "test": "two-sided paired sign-flip permutation, 10000 resamples, seed 0",
        },
    )

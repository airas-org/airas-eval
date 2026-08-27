"""The scoring entry point: ``evaluate(task_type, inputs)``.

Computes the full pinned metric set for a task type — there is no parameter
for choosing metrics, variants, or which part of the task to report. Inputs
are one dict per named group: ``{"main": {...}}``.
What cannot be computed is reported under ``skipped`` with a
machine-readable code, never silently dropped; and ONLY the dedicated skip
exceptions are converted to skips — a plain ValueError (malformed inputs, a
bug in a metric) fails the evaluation.

Note on trust: this library cannot defend itself inside an agent-controlled
process (anything can be monkey-patched there). The trustworthy boundary is
the ``airas-eval score`` CLI run from a pinned environment the agent cannot
edit; the Python API is a convenience for that job and for humans.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata as _metadata
from typing import Any

import numpy as np
from pydantic import ValidationError

from airas_eval import __version__
from airas_eval.exceptions import NotApplicable, UndefinedMetric
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


def _validate_groups(task: TaskSpec, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return ``{group: validated dict | None}`` for every declared group.

    Fail-closed on the contract: unknown groups, missing required groups,
    and any per-group validation error raise ValueError.
    """
    if not isinstance(inputs, dict):
        raise ValueError(
            f"invalid inputs for task {task.task_type!r}: expected a dict of "
            f"input groups {[g.name for g in task.groups]}"
        )
    declared = {g.name for g in task.groups}
    unknown = set(inputs) - declared
    if unknown:
        raise ValueError(
            f"invalid inputs for task {task.task_type!r}: unknown group(s) "
            f"{sorted(unknown)}; declared groups are {sorted(declared)}"
        )
    missing = [g.name for g in task.groups if g.required and inputs.get(g.name) is None]
    if missing:
        raise ValueError(
            f"invalid inputs for task {task.task_type!r}: required group(s) "
            f"{missing} are missing"
        )
    if all(inputs.get(g.name) is None for g in task.groups):
        raise ValueError(
            f"invalid inputs for task {task.task_type!r}: at least one of the "
            f"groups {sorted(declared)} must be provided"
        )
    data: dict[str, Any] = {}
    for group in task.groups:
        raw = inputs.get(group.name)
        if raw is None:
            data[group.name] = None
            continue
        try:
            model = group.bundle.input_model.model_validate(raw)
        except ValidationError as err:
            raise ValueError(
                f"invalid inputs for task {task.task_type!r}, group "
                f"{group.name!r}: {err}"
            ) from err
        data[group.name] = model.model_dump()
    return data


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
    """Compute the full pinned metric set for ``task_type`` on ``inputs``.

    ``inputs`` maps each of the task's group names to that group's inputs.
    Fail-closed on the contract: unknown task types, unknown groups or input
    keys, wrong types, and missing required groups or inputs raise. Metrics
    that do not apply to the given data are reported under ``skipped`` with
    a machine-readable code and a reason; an omitted optional group skips
    all of its metrics with ``missing_optional_input``.
    """
    task = _get_task(task_type)
    data = _validate_groups(task, inputs)

    metrics: dict[str, float] = {}
    curves: dict[str, list[Any]] = {}
    skipped: dict[str, dict[str, str]] = {}
    inputs_summary: dict[str, float] = {}
    omitted: list[str] = []

    for group in task.groups:
        group_data = data[group.name]
        bundle = group.bundle
        if group_data is None:
            omitted.append(group.name)
            for binding in bundle.bindings:
                skipped[f"{group.name}.{binding.name}"] = {
                    "code": SkipCode.MISSING_OPTIONAL_INPUT.value,
                    "reason": f"optional group {group.name!r} was not provided",
                }
            continue
        omitted.extend(
            f"{group.name}.{key}"
            for key in bundle.optional_inputs()
            if group_data.get(key) is None
        )

        for target, bindings in (
            (metrics, bundle.metrics),
            (inputs_summary, bundle.summary),
        ):
            for binding in bindings:
                full_name = f"{group.name}.{binding.name}"
                value, skip = _run_binding(binding, group_data)
                if skip is not None:
                    skipped[full_name] = skip
                    continue
                score = float(value)
                if not np.isfinite(score):
                    skipped[full_name] = {
                        "code": SkipCode.UNDEFINED_ON_DATA.value,
                        "reason": f"non-finite result ({score})",
                    }
                    continue
                target[full_name] = score

        for binding in bundle.curves:
            full_name = f"{group.name}.{binding.name}"
            value, skip = _run_binding(binding, group_data)
            if skip is not None:
                skipped[full_name] = skip
                continue
            curve = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(curve)):
                skipped[full_name] = {
                    "code": SkipCode.UNDEFINED_ON_DATA.value,
                    "reason": "non-finite values in curve",
                }
                continue
            curves[full_name] = curve.tolist()

    return EvaluationReport(
        task_type=task_type,
        metrics=metrics,
        curves=curves,
        skipped=skipped,
        inputs_summary=inputs_summary,
        omitted_optional_inputs=omitted,
        provenance={
            "task_signature": task.signature(),
            "versions": _versions(task.provenance_packages()),
            "inputs_sha256": _inputs_sha256(data),
        },
    )

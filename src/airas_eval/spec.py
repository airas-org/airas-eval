"""Task schema: what a task type is, and how its signature is derived.

Two public layers only:

* ``metrics/`` — pure functions, no task knowledge.
* ``tasks/`` — a task type is one situation ("NAS, before training, look at
  performance") bound to ONE inputs file and the FULL set of metrics that
  situation must report. ``evaluate("nas_pre_training", {...})``.

Internally, tasks are assembled from :class:`MetricSet` constants (e.g. "the
classification metrics") so the same binding is written once and reused by
several tasks. Metric sets are not registered, not evaluable and not shown
to callers — the only thing a caller can choose is the task type.

The task signature is DERIVED from the declaration (never hand-written): a
hash over the task type, its input fields, and every binding's (name,
function qualname, inputs, kwargs). Any change to what gets computed changes
the signature.
"""

import hashlib
import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


def unwrap_text(text: str) -> str:
    """Join wrapped docstring lines: no space where a CJK character meets
    another CJK character, one space everywhere else (so "NAS の" and
    "Brier スコア" keep their spacing across a line break)."""
    words = text.split()
    out = words[:1]
    for word in words[1:]:
        wide = unicodedata.east_asian_width
        if wide(out[-1][-1]) in ("W", "F") and wide(word[0]) in ("W", "F"):
            out[-1] += word
        else:
            out.append(word)
    return " ".join(out)


class SkipCode(str, Enum):
    """Machine-readable reasons a metric was not computed.

    Downstream policy can act on these — e.g. treat a large number of
    ``missing_optional_input`` skips as suspicious (omitting probabilities
    hides every calibration metric), while ``not_applicable`` is benign.
    """

    MISSING_OPTIONAL_INPUT = "missing_optional_input"
    NOT_APPLICABLE = "not_applicable"
    UNDEFINED_ON_DATA = "undefined_on_data"
    MISSING_DEPENDENCY = "missing_dependency"


@dataclass(frozen=True)
class MetricBinding:
    """One reported number: a named function applied to named inputs.

    The function is called as ``fn(*(input values in order), **kwargs)``.
    ``fn`` must be a module-level function (its qualname is hashed into the
    task signature); lambdas are rejected at task construction.
    """

    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Documentation, not part of the signature:
    description: str = ""  # human-readable (Japanese)
    value_range: str = ""  # e.g. "[0, 1]", "[-1, 1]", "[0, ∞)", "スコアと同じ単位"
    direction: str = ""  # "higher" | "lower" | "none" (higher/lower is better)

    def declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function": self.fn.__qualname__,
            "inputs": list(self.inputs),
            "kwargs": dict(sorted(self.kwargs.items())),
        }


@dataclass(frozen=True)
class MetricSet:
    """A reusable, internal group of bindings over agreed input field names.

    ``metrics`` and ``curves`` are the evaluation results; ``summary`` holds
    input-size facts (n_examples, ...) reported apart from metrics;
    ``per_example`` bindings return one score per example (higher is better)
    and exist only to feed paired comparisons between two systems.
    """

    metrics: tuple[MetricBinding, ...]
    curves: tuple[MetricBinding, ...] = ()
    summary: tuple[MetricBinding, ...] = ()
    per_example: tuple[MetricBinding, ...] = ()
    provenance_packages: tuple[str, ...] = ("numpy",)
    notes: str = ""


def _compact_type(schema: dict[str, Any]) -> str:
    """Render a JSON Schema fragment as a short type, e.g. ``number[][]``."""
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        return " | ".join(_compact_type(o) for o in options)
    kind = schema.get("type")
    if kind == "array":
        return _compact_type(schema.get("items", {})) + "[]"
    if kind == "integer":
        return "int"
    return str(kind or "any")


@dataclass(frozen=True)
class TaskSpec:
    """The full metric contract for one task type: one inputs file, all the
    metrics that situation must report. Report keys are the metric names."""

    task_type: str
    input_model: type[BaseModel]
    metrics: tuple[MetricBinding, ...]
    curves: tuple[MetricBinding, ...] = ()
    summary: tuple[MetricBinding, ...] = ()
    per_example: tuple[MetricBinding, ...] = ()
    provenance_packages: tuple[str, ...] = ("numpy",)
    description: str = ""
    notes: str = ""
    schema_version: int = 1

    @classmethod
    def from_sets(
        cls,
        task_type: str,
        input_model: type[BaseModel],
        *sets: MetricSet,
        description: str = "",
        notes: str = "",
    ) -> "TaskSpec":
        """Assemble a task from reusable metric sets (concatenated in order)."""
        if not sets:
            raise ValueError(f"task {task_type!r} needs at least one metric set")
        packages: dict[str, None] = {}
        for s in sets:
            for p in s.provenance_packages:
                packages.setdefault(p, None)
        set_notes = "。".join(s.notes for s in sets if s.notes)
        return cls(
            task_type=task_type,
            input_model=input_model,
            metrics=tuple(b for s in sets for b in s.metrics),
            curves=tuple(b for s in sets for b in s.curves),
            summary=tuple(b for s in sets for b in s.summary),
            per_example=tuple(b for s in sets for b in s.per_example),
            provenance_packages=tuple(packages),
            description=description,
            notes="。".join(n for n in (set_notes, notes) if n),
        )

    def __post_init__(self) -> None:
        if not self.task_type.isidentifier():
            raise ValueError(f"task type {self.task_type!r} must be an identifier")
        if not self.metrics:
            raise ValueError(f"task {self.task_type!r} declares no metrics")
        names = [b.name for b in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate metric names in task {self.task_type!r}")
        model_fields = set(self.input_model.model_fields)
        for binding in self.bindings:
            if not binding.description.strip():
                raise ValueError(f"{binding.name}: every binding needs a description")
        for binding in self.metrics + self.curves:
            if not binding.value_range.strip():
                raise ValueError(f"{binding.name}: every metric needs a value_range")
            if binding.direction not in ("higher", "lower", "none"):
                raise ValueError(
                    f"{binding.name}: direction must be higher, lower or none"
                )
            if "<lambda>" in getattr(binding.fn, "__qualname__", "<lambda>"):
                raise ValueError(
                    f"{binding.name}: bindings must reference named module-level "
                    "functions, not lambdas"
                )
            unknown = set(binding.inputs) - model_fields
            if unknown:
                raise ValueError(
                    f"{self.task_type}/{binding.name} requires inputs "
                    f"{sorted(unknown)} not present in {self.input_model.__name__}"
                )

    @property
    def bindings(self) -> tuple[MetricBinding, ...]:
        return self.metrics + self.curves + self.summary + self.per_example

    def required_inputs(self) -> tuple[str, ...]:
        return tuple(
            name for name, f in self.input_model.model_fields.items() if f.is_required()
        )

    def optional_inputs(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, f in self.input_model.model_fields.items()
            if not f.is_required()
        )

    def input_fields(self) -> list[dict[str, Any]]:
        """Per-input name, JSON type and description, in declaration order.

        Derived from the pydantic model's JSON Schema so ``list`` shows the
        same contract that ``schema`` emits and ``validate`` enforces.
        """
        props = self.input_model.model_json_schema()["properties"]
        return [
            {
                "name": name,
                "type": _compact_type(props[name]),
                "required": f.is_required(),
                "description": f.description or "",
            }
            for name, f in self.input_model.model_fields.items()
        ]

    def describe(self) -> dict[str, Any]:
        """Human/agent-facing description of what this task takes and returns.

        Unlike ``declaration`` this includes descriptions and notes and is
        not hashed into the signature; it is what ``airas-eval list --json``
        prints.
        """

        def rows(kind: str, items: tuple[MetricBinding, ...]) -> list[dict[str, Any]]:
            return [
                {
                    "name": b.name,
                    "kind": kind,
                    "description": b.description,
                    "value_range": b.value_range,
                    "direction": b.direction,
                    "pinned": dict(sorted(b.kwargs.items())),
                    "inputs": list(b.inputs),
                }
                for b in items
            ]

        return {
            "task_type": self.task_type,
            "signature": self.signature(),
            "description": unwrap_text(self.description),
            "notes": self.notes,
            "required_inputs": list(self.required_inputs()),
            "optional_inputs": list(self.optional_inputs()),
            "inputs": self.input_fields(),
            "metrics": rows("scalar", self.metrics) + rows("curve", self.curves),
            "inputs_summary": rows("input_size", self.summary),
            "per_example": rows("per_example", self.per_example),
        }

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema of the inputs file, generated from the same pydantic
        model that validates inputs, so the contract an agent reads and the
        check the evaluator applies cannot diverge."""
        schema = self.input_model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = f"{self.task_type} inputs"
        schema["description"] = unwrap_text(self.description)
        return schema

    def declaration(self) -> dict[str, Any]:
        """Canonical description of what this task computes.

        Everything that affects the reported numbers is in here; the
        signature is a hash of this. Deliberately excludes docstrings, notes
        and module paths, so refactors that cannot change behavior do not
        change the signature.
        """
        return {
            "task_type": self.task_type,
            "schema_version": self.schema_version,
            "required_inputs": list(self.required_inputs()),
            "optional_inputs": list(self.optional_inputs()),
            "metrics": [b.declaration() for b in self.metrics],
            "curves": [b.declaration() for b in self.curves],
            "summary": [b.declaration() for b in self.summary],
            "per_example": [b.declaration() for b in self.per_example],
        }

    def signature(self) -> str:
        canonical = json.dumps(
            self.declaration(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{self.task_type}/v{self.schema_version}@{digest}"

"""Task schema: what a task type is, and how its signature is derived.

Two public layers only:

* ``metrics/`` — pure functions, no task knowledge.
* ``tasks/`` — a task type is the full set of metrics a study of that kind
  must report, declared as named *input groups*. Each group binds a pydantic
  input model to metric bindings; a group is either required or optional.
  ``evaluate("nas_post_training", {"architecture": {...}})``.

The reusable building block between them is a :class:`Bundle` (e.g. "the
classification metrics"). Bundles are plain module constants: they are not
registered and cannot be evaluated on their own, so the only thing a caller
can choose is the task type — never which part of it to report. An optional
group that is omitted is surfaced in the report like any omitted optional
input.

The task signature is DERIVED from the declaration (never hand-written): a
hash over the task type, every group's name / requiredness / input fields,
and every binding's (name, function qualname, inputs, kwargs). Any change to
what gets computed changes the signature.
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
    """Join wrapped docstring lines: no space between CJK characters, one
    space otherwise (so English words at a line break stay separated)."""
    words = text.split()
    out = words[:1]
    for word in words[1:]:
        wide = unicodedata.east_asian_width
        if wide(out[-1][-1]) in ("W", "F") or wide(word[0]) in ("W", "F"):
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
    task signature); lambdas are rejected at bundle construction.
    """

    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""  # human-readable (Japanese); not part of the signature

    def declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function": self.fn.__qualname__,
            "inputs": list(self.inputs),
            "kwargs": dict(sorted(self.kwargs.items())),
        }


@dataclass(frozen=True)
class Bundle:
    """A reusable set of metrics over one input model.

    ``metrics`` and ``curves`` are the evaluation results; ``summary`` holds
    input-size facts (n_examples, ...) that are reported separately from
    metrics but are part of the contract and the signature all the same.
    ``per_example`` bindings return one score per example (higher is better)
    and exist only to feed paired comparisons between two systems.

    Not registered, not evaluable on its own: tasks compose bundles into
    named groups. Validation of the declaration happens here so a broken
    bundle fails at import time.
    """

    input_model: type[BaseModel]
    metrics: tuple[MetricBinding, ...]
    curves: tuple[MetricBinding, ...] = ()
    summary: tuple[MetricBinding, ...] = ()
    per_example: tuple[MetricBinding, ...] = ()
    provenance_packages: tuple[str, ...] = ("numpy",)
    notes: str = ""

    def __post_init__(self) -> None:
        names = [b.name for b in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError(
                f"duplicate metric names in bundle over {self.input_model.__name__}"
            )
        model_fields = set(self.input_model.model_fields)
        for binding in self.bindings:
            if not binding.description.strip():
                raise ValueError(f"{binding.name}: every binding needs a description")
            if "<lambda>" in getattr(binding.fn, "__qualname__", "<lambda>"):
                raise ValueError(
                    f"{binding.name}: bindings must reference named module-level "
                    "functions, not lambdas"
                )
            unknown = set(binding.inputs) - model_fields
            if unknown:
                raise ValueError(
                    f"{binding.name} requires inputs {sorted(unknown)} not present "
                    f"in {self.input_model.__name__}"
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

    def declaration(self) -> dict[str, Any]:
        return {
            "required_inputs": list(self.required_inputs()),
            "optional_inputs": list(self.optional_inputs()),
            "metrics": [b.declaration() for b in self.metrics],
            "curves": [b.declaration() for b in self.curves],
            "summary": [b.declaration() for b in self.summary],
            "per_example": [b.declaration() for b in self.per_example],
        }


@dataclass(frozen=True)
class Group:
    """One named input group of a task: a bundle plus whether it is required."""

    name: str
    bundle: Bundle
    required: bool = True

    def declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            **self.bundle.declaration(),
        }


@dataclass(frozen=True)
class TaskSpec:
    """The full metric contract for one task type: an ordered set of groups.

    Metric names in the report are ``<group>.<metric>``. A task may declare
    only optional groups; the evaluator then requires at least one of them.
    """

    task_type: str
    groups: tuple[Group, ...]
    description: str = ""
    notes: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError(f"task {self.task_type!r} declares no groups")
        names = [g.name for g in self.groups]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate group names in task {self.task_type!r}")
        for name in names:
            if not name.isidentifier():
                raise ValueError(f"group name {name!r} must be an identifier")

    def group(self, name: str) -> Group:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(name)

    def required_groups(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups if g.required)

    def optional_groups(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups if not g.required)

    def provenance_packages(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for g in self.groups:
            for package in g.bundle.provenance_packages:
                seen.setdefault(package, None)
        return tuple(seen)

    def describe(self) -> dict[str, Any]:
        """Human/agent-facing description of what this task returns.

        Unlike ``declaration`` this includes descriptions and notes and is
        not hashed into the signature; it is what ``airas-eval list --json``
        prints.
        """

        def bindings(
            group: Group, kind: str, items: tuple[MetricBinding, ...]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "name": f"{group.name}.{b.name}",
                    "kind": kind,
                    "description": b.description,
                    "pinned": dict(sorted(b.kwargs.items())),
                    "inputs": list(b.inputs),
                }
                for b in items
            ]

        return {
            "task_type": self.task_type,
            "signature": self.signature(),
            "description": unwrap_text(self.description),
            "groups": [
                {
                    "name": g.name,
                    "required": g.required,
                    "notes": g.bundle.notes,
                    "required_inputs": list(g.bundle.required_inputs()),
                    "optional_inputs": list(g.bundle.optional_inputs()),
                    "metrics": bindings(g, "scalar", g.bundle.metrics)
                    + bindings(g, "curve", g.bundle.curves),
                    "inputs_summary": bindings(g, "input_size", g.bundle.summary),
                    "per_example": bindings(g, "per_example", g.bundle.per_example),
                }
                for g in self.groups
            ],
        }

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema of the inputs file: one object per group.

        Generated from the same pydantic models that validate inputs, so the
        contract an agent reads and the check the evaluator applies cannot
        diverge. Semantic conventions live in the field descriptions.
        """
        properties: dict[str, Any] = {}
        for group in self.groups:
            schema = group.bundle.input_model.model_json_schema()
            schema["description"] = group.bundle.notes
            properties[group.name] = schema
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{self.task_type} inputs",
            "description": unwrap_text(self.description),
            "type": "object",
            "properties": properties,
            "required": list(self.required_groups()),
            "minProperties": 1,
            "additionalProperties": False,
        }

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
            "groups": [g.declaration() for g in self.groups],
        }

    def signature(self) -> str:
        canonical = json.dumps(
            self.declaration(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{self.task_type}/v{self.schema_version}@{digest}"

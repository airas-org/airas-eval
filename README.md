# airas-eval

The trusted evaluation layer for [AIRAS](https://github.com/airas-org/airas):
agents pass a task type and raw predictions, and get the full fixed set of
standard metrics back. Agents never implement evaluation scripts and never
choose which metrics (or which variants) get reported.

## Structure

Two layers, and nothing a caller can choose between them:

1. **`metrics/`** — metric implementations, one module per input shape, no
   task knowledge. Delegated to scikit-learn / scipy wherever a canonical
   implementation exists; in-house only where none does (ECE, regret,
   top-k selection, 2-D hypervolume, IGD/GD/spacing), pinned and
   property-tested.
2. **`tasks/`** — one sub-package per area, one module per task type. A task
   type is the full set of metrics a study of that kind must report (5–12
   of them, scalar or curve, each with its variant pinned), computed from
   one validated input group. `generic/` holds the core evaluation families; an area package
   (`nas/`) builds on them — its bundles are *the core bindings plus what
   that area's literature adds*, never a renamed copy. The reusable pieces
   (`tasks/_bundles.py`, `tasks/nas/_bundles.py`) are plain constants: not
   registered, not evaluable on their own.

```
tasks/
├── generic/   classification, binary_classification, search, candidate_ranking, multiobjective
└── nas/       nas_search       = search + wall-clock axis, search-space position,
               │                  random-search baseline, test regret
               ├── nas_architecture = classification + random-architecture baseline (Yang et al. 2020)
               └── nas_predictor    = candidate_ranking + top-10% rank correlation (NAS-Bench-Suite-Zero)
```

## What does each task return?

Two sources, both derived from the registry so they cannot drift from
behavior:

```bash
airas-eval list                # every task type: inputs, metrics, curves, summary, signature
airas-eval list nas_search     # one task type
```

and a generated README per area, checked by the test suite:
[`tasks/generic/README.md`](src/airas_eval/tasks/generic/README.md),
[`tasks/nas/README.md`](src/airas_eval/tasks/nas/README.md).
Regenerate with `python -m airas_eval.tasks.readme` after changing a task or
bundle.

## Usage

```python
from airas_eval import evaluate

report = evaluate(
    "nas_search",
    {"main": {"evaluated_scores": scores_in_order, "oracle_best": 94.37}},
)
report.metrics  # scalar metrics: {"main.best_score": ..., "main.final_regret": ..., ...}
report.curves  # non-scalar metrics: {"main.best_so_far": [...]}
report.inputs_summary  # NOT metrics — input sizes: {"main.n_evaluations": 200}
report.skipped  # what didn't apply — machine-readable code + reason
report.omitted_optional_inputs  # e.g. ["main.oracle_best"]
report.provenance  # derived task signature, versions, input SHA-256
```

```bash
airas-eval score nas_search --inputs inputs.json --output evaluation.json
```

`examples/` holds a minimal input file per NAS task; the test suite scores
each of them through the CLI. NAS-specific inputs (`evaluation_costs`,
`search_space_scores`, `random_architecture_accuracies`, ...) are optional
reference data: leave them out and the metrics that need them are reported
as skipped, with the omission listed.

Inputs are always grouped (`{"main": {...}}`); a task type is the only thing
the caller chooses, and which task type a study is evaluated as belongs to
the research plan, not to the evaluation step.

## Design rules

1. **No metric choice.** A task type computes *all* its metrics; the set is
   fixed and small enough to read, because a long list is its own kind of
   cherry-picking surface. Variants (averaging, k, bins) are pinned per
   task, and every parameter a metric function takes must be pinned
   explicitly (enforced by test). Input sizes (`n_examples`,
   `n_evaluations`, ...) are reported under `inputs_summary`, apart from
   metrics, so a subsetted test set or a truncated run is visible.
2. **Nothing disappears silently.** A metric that cannot be computed appears
   under `skipped` with a machine-readable code (`missing_optional_input`,
   `not_applicable`, `undefined_on_data`, `missing_dependency`); omitted
   optional inputs are surfaced per report.
   Only these dedicated cases become skips — malformed inputs and library
   bugs raise instead of hiding as "undefined".
3. **Provenance is derived, never hand-written.** The task signature is a
   hash of the task's declaration (metric names, function identities,
   pinned kwargs, input fields), so it cannot drift from behavior. The
   per-area READMEs are generated from the same declaration.
4. **Reference data is fixed upstream.** `reference_labels`, `oracle_best`,
   and reference points/fronts belong to the experimental design. This
   library cannot verify they are genuine, and cannot defend itself inside
   an agent-controlled process — run `airas-eval score` from a pinned
   environment the agent cannot edit.

## Install

Not on PyPI yet. Pin a tag so the evaluation layer cannot change under a
study:

```bash
uvx --from git+https://github.com/airas-org/airas-eval@v0.2.0 airas-eval list
uv add "airas-eval @ git+https://github.com/airas-org/airas-eval@v0.2.0"
```

Dependencies: numpy, scikit-learn, scipy, pydantic.

## Development

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Adding a task type = one module under `tasks/<area>/` defining `TASK` from
an existing bundle, one line in the area's `TASKS`, then
`python -m airas_eval.tasks.readme`. A new area = a new sub-package plus one
line in `tasks.AREAS`. Adding a bundle = an input model in `tasks/_inputs.py`
and a `Bundle` in `tasks/_bundles.py` with a `summary` count. Keep tasks at
roughly 5–10 metrics: standard variants only, one pin per parameter.

Registration is deliberately explicit (no entry points, no scanning): in a
trust layer, what gets computed must be visible in a reviewed diff. New
in-house metrics need parity tests against an oracle implementation, or
hand-computed cases plus property tests.

## License

MIT

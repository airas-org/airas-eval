"""CLI: the consumer boundary of airas-eval.

Agents interact with airas-eval only through files and this CLI:

* ``list``      what each task type computes (fixed; nothing to choose)
* ``schema``    the JSON Schema of a task type's inputs file (what to produce)
* ``validate``  check an inputs file against that contract, without scoring
* ``score``     compute the full metric set for one inputs file
* ``aggregate`` mean ± std over several reports (e.g. seeds)
* ``compare``   paired significance test between two inputs files

``score`` is the process-level trust boundary: run it from a pinned
environment the agent cannot edit. Which metrics run, and with which
variants, is decided by the task type — there are deliberately no flags for
that.
"""

import argparse
import json
import sys
from typing import Any

from airas_eval import (
    __version__,
    aggregate_reports,
    compare,
    evaluate,
    validate_inputs,
)
from airas_eval.tasks import TASKS


def _load(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _emit(payload: str, output: str | None) -> None:
    if output:
        with open(output, "w") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


def _print_list(only: str | None) -> None:
    for task_type, task in sorted(TASKS.items()):
        if only and task_type != only:
            continue

        print(f"{task_type}:  [{task.signature()}]")
        for group in task.groups:
            bundle = group.bundle
            kind = "required" if group.required else "optional"
            print(f"  {group.name}  ({kind})")
            print(f"    required inputs: {', '.join(bundle.required_inputs())}")
            if bundle.optional_inputs():
                print(f"    optional inputs: {', '.join(bundle.optional_inputs())}")
            for binding in bundle.metrics:
                print(f"      {group.name}.{binding.name}")
            for binding in bundle.curves:
                print(f"      {group.name}.{binding.name}  [curve]")
            if bundle.summary:
                sizes = ", ".join(f"{group.name}.{b.name}" for b in bundle.summary)
                print(f"    input sizes (not metrics): {sizes}")
            if bundle.per_example:
                names = ", ".join(f"{group.name}.{b.name}" for b in bundle.per_example)
                print(f"    per-example scores (for compare): {names}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="airas-eval")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list task types and their metrics")
    p_list.add_argument("task_type", nargs="?", help="show one task type only")

    p_schema = sub.add_parser("schema", help="JSON Schema of a task type's inputs file")
    p_schema.add_argument("task_type", choices=sorted(TASKS))
    p_schema.add_argument("--output", help="write the schema here (default: stdout)")

    p_validate = sub.add_parser(
        "validate", help="check an inputs file, without scoring"
    )
    p_validate.add_argument("task_type", choices=sorted(TASKS))
    p_validate.add_argument("--inputs", required=True, help="path to {group: inputs}")

    p_score = sub.add_parser(
        "score", help="compute the full metric set for a task type"
    )
    p_score.add_argument("task_type", choices=sorted(TASKS))
    p_score.add_argument("--inputs", required=True, help="path to {group: inputs}")
    p_score.add_argument(
        "--output", help="write the report JSON here (default: stdout)"
    )

    p_agg = sub.add_parser("aggregate", help="mean ± std over several score reports")
    p_agg.add_argument("--reports", nargs="+", required=True, help="report JSON files")
    p_agg.add_argument(
        "--output", help="write the aggregate JSON here (default: stdout)"
    )

    p_cmp = sub.add_parser("compare", help="paired significance test, system A vs B")
    p_cmp.add_argument("task_type", choices=sorted(TASKS))
    p_cmp.add_argument("--a", required=True, help="inputs file of system A")
    p_cmp.add_argument("--b", required=True, help="inputs file of system B")
    p_cmp.add_argument(
        "--output", help="write the comparison JSON here (default: stdout)"
    )

    args = parser.parse_args()

    if args.command == "list":
        _print_list(args.task_type)
    elif args.command == "schema":
        schema = TASKS[args.task_type].input_schema()
        _emit(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", args.output)
    elif args.command == "validate":
        try:
            groups = validate_inputs(args.task_type, _load(args.inputs))
        except ValueError as err:
            sys.stderr.write(f"INVALID: {err}\n")
            sys.exit(1)
        print(
            f"OK: {args.task_type} inputs valid; groups provided: {', '.join(groups)}"
        )
    elif args.command == "score":
        report = evaluate(args.task_type, _load(args.inputs))
        _emit(report.to_json() + "\n", args.output)
    elif args.command == "aggregate":
        aggregate = aggregate_reports([_load(path) for path in args.reports])
        _emit(aggregate.to_json() + "\n", args.output)
    elif args.command == "compare":
        result = compare(args.task_type, _load(args.a), _load(args.b))
        _emit(result.to_json() + "\n", args.output)


if __name__ == "__main__":
    main()

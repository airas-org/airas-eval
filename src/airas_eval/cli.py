"""CLI: the consumer boundary of airas-eval.

Agents interact with airas-eval only through files and this CLI:

* ``list``      what each task type computes (fixed; nothing to choose);
                ``--json`` for the machine-readable contract
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


def _short(text: str, limit: int = 45) -> str:
    """Leading sentences of a description, up to roughly ``limit`` characters."""
    out = ""
    for sentence in text.split("。"):
        if not sentence:
            continue
        if out and len(out) + len(sentence) > limit:
            break
        out += sentence + "。"
    return out


_KIND_HEADERS = (
    ("metrics", "指標"),
    ("inputs_summary", "入力サイズ(指標ではない)"),
    ("per_example", "事例ごとのスコア(compare 用)"),
)


def _print_list(only: str | None) -> None:
    for task_type, task in sorted(TASKS.items()):
        if only and task_type != only:
            continue
        info = task.describe()
        print(f"{task_type}:  [{info['signature']}]")
        if info["description"]:
            print(f"  {_short(info['description'], 80)}")
        for group in info["groups"]:
            kind = "必須" if group["required"] else "任意"
            print(f"\n  グループ {group['name']}({kind})")
            print(f"    必須入力: {', '.join(group['required_inputs']) or '—'}")
            print(f"    任意入力: {', '.join(group['optional_inputs']) or '—'}")
            rows = group["metrics"] + group["inputs_summary"] + group["per_example"]
            width = max(len(r["name"]) for r in rows)
            for key, header in _KIND_HEADERS:
                if not group[key]:
                    continue
                print(f"    {header}:")
                for r in group[key]:
                    tag = "  (曲線)" if r["kind"] == "curve" else ""
                    pinned = (
                        "  ["
                        + ", ".join(f"{k}={v}" for k, v in r["pinned"].items())
                        + "]"
                        if r["pinned"]
                        else ""
                    )
                    print(
                        f"      {r['name']:<{width}}  {_short(r['description'])}"
                        f"{pinned}{tag}"
                    )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="airas-eval")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list task types and their metrics")
    p_list.add_argument("task_type", nargs="?", help="show one task type only")
    p_list.add_argument(
        "--json", action="store_true", help="machine-readable: {task_type: contract}"
    )

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
        if args.json:
            selected = {
                t: TASKS[t].describe()
                for t in sorted(TASKS)
                if not args.task_type or t == args.task_type
            }
            sys.stdout.write(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
        else:
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

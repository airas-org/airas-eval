"""CLI: inspect task types and score input files.

``airas-eval score`` is the process-level trust boundary: run it from a
pinned environment the agent cannot edit, with inputs from a JSON file, and
the full report goes to stdout or a file. Which metrics run, and with
which variants, is decided by the task type — there are deliberately no flags
for that.
"""

import argparse
import json
import sys

from airas_eval import __version__, evaluate
from airas_eval.tasks import TASKS


def main() -> None:
    parser = argparse.ArgumentParser(prog="airas-eval")
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list", help="list task types and their metrics")
    list_parser.add_argument("task_type", nargs="?", help="show one task type only")

    score_parser = sub.add_parser(
        "score", help="compute the full metric set for a task type"
    )
    score_parser.add_argument("task_type", choices=sorted(TASKS))
    score_parser.add_argument(
        "--inputs", required=True, help="path to a JSON file: {group: inputs}"
    )
    score_parser.add_argument(
        "--output", help="write the report JSON here (default: stdout)"
    )

    args = parser.parse_args()

    if args.command == "list":
        for task_type, task in sorted(TASKS.items()):
            if args.task_type and task_type != args.task_type:
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
                for binding in bundle.summary:
                    print(f"      {group.name}.{binding.name}  [summary]")
        return

    with open(args.inputs) as f:
        inputs = json.load(f)

    report = evaluate(args.task_type, inputs)
    payload = report.to_json() + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()

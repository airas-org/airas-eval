"""CLI: inspect suites and score input files.

``airas-eval score`` is the process-level entry point for trusted scoring
jobs: inputs come from a JSON file, the full suite report goes to stdout or a
file. Which metrics run, and with which variants, is decided by the suite —
there are deliberately no flags for that.
"""

import argparse
import json
import sys

from airas_eval import __version__, evaluate
from airas_eval.suite import SUITES


def main() -> None:
    parser = argparse.ArgumentParser(prog="airas-eval")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list task types and their metric suites")
    list_parser.add_argument("task_type", nargs="?", help="show one task type only")

    score_parser = sub.add_parser("score", help="compute the full suite for a task")
    score_parser.add_argument("task_type", choices=sorted(SUITES))
    score_parser.add_argument(
        "--inputs", required=True, help="path to a JSON file with the suite inputs"
    )
    score_parser.add_argument(
        "--output", help="write the report JSON here (default: stdout)"
    )

    args = parser.parse_args()

    if args.command == "list":
        for task_type, suite in sorted(SUITES.items()):
            if args.task_type and task_type != args.task_type:
                continue
            print(f"{task_type}:")
            print(f"  required inputs: {', '.join(suite.required_inputs)}")
            if suite.optional_inputs:
                print(f"  optional inputs: {', '.join(suite.optional_inputs)}")
            for metric in suite.metrics:
                print(f"    {metric.name}")
        return

    with open(args.inputs) as f:
        inputs = json.load(f)
    report = evaluate(args.task_type, inputs)
    payload = report.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()

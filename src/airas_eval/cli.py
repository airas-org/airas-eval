"""Minimal CLI: inspect the metric registry."""

import argparse

from airas_eval import __version__
from airas_eval.registry import REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(prog="airas-eval")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list", help="list registered metrics")
    list_parser.add_argument("task_type", nargs="?", help="filter by task type")
    args = parser.parse_args()

    if args.command == "list":
        for task_type, specs in sorted(REGISTRY.items()):
            if args.task_type and task_type != args.task_type:
                continue
            print(f"{task_type}:")
            for spec in specs:
                origin = (
                    f"wraps {spec.wrapped_package}" if spec.wrapped_package else "core"
                )
                print(f"  {spec.name:28s} [{origin}]")


if __name__ == "__main__":
    main()

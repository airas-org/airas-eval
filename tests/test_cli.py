"""The CLI is the consumer boundary (``uvx airas-eval score``); test it as a
subprocess, not through the Python API."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from airas_eval.tasks import TASKS

EXAMPLES = Path(__file__).parent.parent / "examples"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "airas_eval.cli", *args],
        capture_output=True,
        text=True,
    )


def test_list_prints_every_registered_task():
    out = _run("list")
    assert out.returncode == 0, out.stderr
    for task_type in TASKS:
        assert f"{task_type}:  [{task_type}/v" in out.stdout


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_examples_score_end_to_end(path: Path, tmp_path: Path):
    task_type = path.stem
    out = _run("score", task_type, "--inputs", str(path))
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["task_type"] == task_type
    assert report["provenance"]["task_signature"] == TASKS[task_type].signature()
    assert report["metrics"], "no metrics computed"
    assert report["inputs_summary"]
    # --output writes the same payload to a file
    target = tmp_path / "report.json"
    assert (
        _run(
            "score", task_type, "--inputs", str(path), "--output", str(target)
        ).returncode
        == 0
    )
    assert json.loads(target.read_text()) == report


def test_every_nas_task_has_an_example():
    missing = [
        t
        for t in TASKS
        if t.startswith("nas_") and not (EXAMPLES / f"{t}.json").exists()
    ]
    assert missing == []


def test_invalid_inputs_fail_nonzero(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"evaluated_scores": [1.0]}))  # ungrouped
    out = _run("score", "nas_pre_training", "--inputs", str(bad))
    assert out.returncode != 0
    assert "unknown group" in out.stderr
    assert _run("score", "no_such_task", "--inputs", str(bad)).returncode != 0

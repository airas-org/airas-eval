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
    bad.write_text(json.dumps({"evaluated_scores": [1.0], "typo": 1}))
    out = _run("score", "nas_pre_training", "--inputs", str(bad))
    assert out.returncode != 0
    assert "invalid inputs" in out.stderr
    assert _run("score", "no_such_task", "--inputs", str(bad)).returncode != 0


def test_schema_and_validate(tmp_path: Path):
    out = _run("schema", "nas_post_training")
    assert out.returncode == 0, out.stderr
    schema = json.loads(out.stdout)
    assert schema["required"] == ["predicted_labels", "reference_labels"]
    assert schema["additionalProperties"] is False
    ok = _run(
        "validate",
        "nas_post_training",
        "--inputs",
        str(EXAMPLES / "nas_post_training.json"),
    )
    assert ok.returncode == 0 and ok.stdout.startswith("OK:"), ok.stderr
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"predicted_labels": [0]}))
    res = _run("validate", "nas_post_training", "--inputs", str(bad))
    assert res.returncode == 1 and res.stderr.startswith("INVALID:")


def test_aggregate_and_compare(tmp_path: Path):
    example = EXAMPLES / "nas_post_training.json"
    reports = []
    for i in range(2):
        target = tmp_path / f"r{i}.json"
        assert (
            _run(
                "score",
                "nas_post_training",
                "--inputs",
                str(example),
                "--output",
                str(target),
            ).returncode
            == 0
        )
        reports.append(str(target))
    agg = _run("aggregate", "--reports", *reports)
    assert agg.returncode == 0, agg.stderr
    payload = json.loads(agg.stdout)
    assert payload["n_reports"] == 2
    assert payload["metrics"]["accuracy"]["std"] == 0.0
    cmp = _run("compare", "nas_post_training", "--a", str(example), "--b", str(example))
    assert cmp.returncode == 0, cmp.stderr
    assert json.loads(cmp.stdout)["comparisons"]["correct"]["mean_diff"] == 0.0


def test_list_json_is_the_full_contract():
    out = _run("list", "--json", "nas_post_training")
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    task = payload["nas_post_training"]
    assert task["signature"] == TASKS["nas_post_training"].signature()
    names = {m["name"] for m in task["metrics"]}
    assert {"accuracy", "test_regret", "hypervolume_2d"} <= names
    assert all(m["description"] for m in task["metrics"])
    assert {m["kind"] for m in task["metrics"]} == {"scalar", "curve"}
    assert task["per_example"][0]["name"] == "correct"
    assert task["required_inputs"] == ["predicted_labels", "reference_labels"]
    inputs = {i["name"]: i for i in task["inputs"]}
    assert inputs["predicted_labels"] == {
        "name": "predicted_labels",
        "type": "int[]",
        "required": True,
        "description": inputs["predicted_labels"]["description"],
    }
    assert inputs["probabilities"]["type"] == "number[][]"
    assert inputs["probabilities"]["required"] is False


def test_list_human_output_has_descriptions():
    out = _run("list", "nas_pre_training")
    assert out.returncode == 0, out.stderr
    assert "入力:" in out.stdout and "指標:" in out.stdout
    assert "best_so_far " in out.stdout and "(曲線)" in out.stdout
    assert "evaluated_scores?: number[]" in out.stdout  # type, `?` marks optional
    assert "oracle_best?: number" in out.stdout and "[任意]" not in out.stdout
    post = _run("list", "nas_post_training").stdout
    assert "predicted_labels: int[]" in post and "probabilities?: number[][]" in post
    assert "fraction=0.1" in out.stdout  # pinned parameters are visible

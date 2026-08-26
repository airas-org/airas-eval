import json

import numpy as np
import pytest

from airas_eval import evaluate
from airas_eval.tasks import TASKS

rng = np.random.default_rng(5)
N = 200
Y_TRUE_MC = rng.integers(0, 6, size=N).tolist()
Y_PRED_MC = np.where(
    rng.random(N) < 0.7, Y_TRUE_MC, rng.integers(0, 6, size=N)
).tolist()
PROBS_MC = rng.dirichlet(np.ones(6), size=N).tolist()
Y_TRUE_BIN = rng.integers(0, 2, size=N).tolist()
Y_PRED_BIN = np.where(
    rng.random(N) < 0.8, Y_TRUE_BIN, 1 - np.array(Y_TRUE_BIN)
).tolist()
PROBS_BIN = np.column_stack(
    [1 - (p := np.clip(np.array(Y_TRUE_BIN) * 0.5 + rng.random(N) * 0.5, 0, 1)), p]
).tolist()

CLS_FULL = {
    "predicted_labels": Y_PRED_MC,
    "reference_labels": Y_TRUE_MC,
    "probabilities": PROBS_MC,
}
CLS_NO_PROBS = {"predicted_labels": Y_PRED_MC, "reference_labels": Y_TRUE_MC}
BIN_FULL = {
    "predicted_labels": Y_PRED_BIN,
    "reference_labels": Y_TRUE_BIN,
    "probabilities": PROBS_BIN,
}
SEARCH_FULL = {"evaluated_scores": [88.0, 91.2, 90.0, 93.1], "oracle_best": 94.37}
RANK_SMALL = {"predicted_scores": [1.0, 2.0, 3.0], "reference_scores": [1.0, 2.0, 3.0]}
MOO = {"points": [[0.1, 5.0], [0.2, 2.0], [0.3, 4.0]], "reference_point": [1.0, 10.0]}


def test_multiclass_full_inputs():
    report = evaluate("classification", {"main": CLS_FULL})
    assert set(report.metrics) == {
        "main.accuracy",
        "main.precision_macro",
        "main.recall_macro",
        "main.f1_macro",
        "main.balanced_accuracy",
        "main.matthews_corrcoef",
        "main.log_loss",
        "main.expected_calibration_error",
        "main.top_5_accuracy",
    }
    assert report.inputs_summary == {"main.n_examples": N, "main.n_classes": 6}
    assert report.skipped == {}
    assert report.omitted_optional_inputs == []


def test_missing_optional_input_is_coded_and_surfaced():
    report = evaluate("classification", {"main": CLS_NO_PROBS})
    assert "main.accuracy" in report.metrics
    assert report.skipped["main.log_loss"]["code"] == "missing_optional_input"
    assert report.skipped["main.n_classes"]["code"] == "missing_optional_input"
    assert report.omitted_optional_inputs == ["main.probabilities"]


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_binary_classification_task():
    report = evaluate("binary_classification", {"main": BIN_FULL})
    assert 0.5 < report.metrics["main.auroc"] <= 1.0
    for name in ("precision", "recall", "f1", "brier_score", "average_precision"):
        assert f"main.{name}" in report.metrics
    assert report.skipped == {}
    # a binary task fed one-class references: data problem, not a shape problem
    one_class = evaluate(
        "binary_classification",
        {
            "main": {
                "predicted_labels": [0, 1, 0],
                "reference_labels": [0, 0, 0],
                "probabilities": [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]],
            }
        },
    )
    assert one_class.skipped["main.auroc"]["code"] == "undefined_on_data"


def test_multiclass_task_marks_top5_not_applicable_for_few_classes():
    report = evaluate("classification", {"main": BIN_FULL})
    assert report.skipped["main.top_5_accuracy"]["code"] == "not_applicable"


def test_contract_is_fail_closed():
    with pytest.raises(KeyError):
        evaluate("no-such-task", {})
    with pytest.raises(ValueError, match="unknown group"):
        evaluate("classification", CLS_NO_PROBS)  # ungrouped
    with pytest.raises(ValueError, match="required group"):
        evaluate("classification", {})
    with pytest.raises(ValueError, match="unknown group"):
        evaluate("classification", {"main": CLS_NO_PROBS, "extra": {}})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "classification",
            {"main": {"predicted_labels": [0], "reference_labels": [0], "typo": [0]}},
        )
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {"main": {"predicted_labels": [0, 1]}})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "candidate_ranking",
            {"main": {"predicted_scores": [float("nan")], "reference_scores": [1.0]}},
        )


def test_malformed_data_raises_instead_of_skipping():
    # A length mismatch is a contract violation, not "undefined on this
    # data" — the evaluator must not swallow it into `skipped`.
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate(
            "candidate_ranking",
            {"main": {"predicted_scores": [1.0, 2.0], "reference_scores": [1.0]}},
        )


def test_search_task():
    report = evaluate("search", {"main": SEARCH_FULL})
    assert report.metrics["main.best_score"] == pytest.approx(93.1)
    assert report.metrics["main.final_regret"] == pytest.approx(1.27)
    assert report.metrics["main.evaluations_to_best"] == 4
    assert report.metrics["main.mean_evaluated_score"] == pytest.approx(90.575)
    assert report.curves["main.best_so_far"] == [88.0, 91.2, 91.2, 93.1]
    assert report.inputs_summary == {"main.n_evaluations": 4}
    without_oracle = evaluate("search", {"main": {"evaluated_scores": [88.0, 91.2]}})
    assert (
        without_oracle.skipped["main.final_regret"]["code"] == "missing_optional_input"
    )
    assert without_oracle.omitted_optional_inputs == ["main.oracle_best"]
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        evaluate("search", {"main": {"evaluated_scores": [95.0], "oracle_best": 94.37}})


def test_candidate_ranking_task():
    reference = [float(i) for i in range(40)]
    predicted = [v + (5.0 if i % 7 == 0 else 0.0) for i, v in enumerate(reference)]
    report = evaluate(
        "candidate_ranking",
        {"main": {"predicted_scores": predicted, "reference_scores": reference}},
    )
    assert 0.0 < report.metrics["main.kendall_tau"] <= 1.0
    assert "main.precision_at_top_10pct" in report.metrics
    assert "main.best_true_rank_in_top_10" in report.metrics
    assert report.inputs_summary == {"main.n_candidates": 40}
    small = evaluate("candidate_ranking", {"main": RANK_SMALL})
    assert small.skipped["main.best_true_rank_in_top_10"]["code"] == "undefined_on_data"
    assert small.skipped["main.precision_at_top_10pct"]["code"] == "undefined_on_data"


def test_multiobjective_task():
    report = evaluate("multiobjective", {"main": MOO})
    assert report.metrics["main.pareto_front_size"] == 2.0
    assert report.metrics["main.hypervolume_2d"] == pytest.approx(6.9)
    assert report.metrics["main.spacing"] == pytest.approx(0.0)  # 2-point front
    assert report.curves["main.pareto_front"] == [[0.1, 5.0], [0.2, 2.0]]
    assert report.inputs_summary == {"main.n_points": 3, "main.n_objectives": 2}
    for name in ("igd", "gd"):
        assert report.skipped[f"main.{name}"]["code"] == "missing_optional_input"
    assert report.omitted_optional_inputs == ["main.reference_front"]


def test_undefined_metric_is_skipped_with_code():
    report = evaluate(
        "candidate_ranking",
        {
            "main": {
                "predicted_scores": [1.0, 1.0, 1.0],
                "reference_scores": [1.0, 2.0, 3.0],
            }
        },
    )
    assert report.skipped["main.kendall_tau"]["code"] == "undefined_on_data"


def test_nas_tasks_share_contracts_with_generic_tasks():
    pairs = {
        "nas_search": ("search", SEARCH_FULL),
        "nas_architecture": ("classification", CLS_FULL),
        "nas_predictor": ("candidate_ranking", RANK_SMALL),
        "nas_tradeoff": ("multiobjective", MOO),
    }
    for nas_task, (generic_task, inputs) in pairs.items():
        a = evaluate(nas_task, {"main": inputs})
        b = evaluate(generic_task, {"main": inputs})
        assert a.metrics == b.metrics
        assert a.skipped == b.skipped
        assert a.inputs_summary == b.inputs_summary
        assert a.provenance["task_signature"].startswith(f"{nas_task}/v1@")
        assert a.provenance["task_signature"] != b.provenance["task_signature"]


def test_provenance_signature_and_hashes():
    inputs = {"main": {"predicted_labels": [0, 1], "reference_labels": [0, 1]}}
    a = evaluate("classification", inputs)
    b = evaluate("classification", inputs)
    c = evaluate(
        "classification",
        {"main": {"predicted_labels": [1, 1], "reference_labels": [0, 1]}},
    )
    assert a.provenance["inputs_sha256"] == b.provenance["inputs_sha256"]
    assert a.provenance["inputs_sha256"] != c.provenance["inputs_sha256"]
    assert a.provenance["task_signature"].startswith("classification/v1@")
    assert a.provenance["versions"]["scikit-learn"] not in ("", "not installed")
    payload = json.loads(a.to_json())
    assert set(payload) == {
        "task_type",
        "metrics",
        "curves",
        "skipped",
        "inputs_summary",
        "omitted_optional_inputs",
        "provenance",
    }


def test_numpy_inputs_are_accepted_and_hashable():
    report = evaluate(
        "classification",
        {
            "main": {
                "predicted_labels": np.array([0, 1, 1]),
                "reference_labels": np.array([0, 1, 0]),
            }
        },
    )
    assert len(report.provenance["inputs_sha256"]) == 64


def test_every_task_reports_input_sizes_outside_metrics():
    for task_type, task in TASKS.items():
        for group in task.groups:
            summary = [b.name for b in group.bundle.summary]
            assert summary and all(n.startswith("n_") for n in summary), task_type
            assert not any(b.name.startswith("n_") for b in group.bundle.metrics), (
                task_type
            )


def test_every_task_is_between_three_and_eleven_metrics():
    for task_type, task in TASKS.items():
        n = sum(len(g.bundle.metrics) for g in task.groups)
        assert 3 <= n <= 11, (task_type, n)


def test_version_comes_from_installed_metadata():
    from importlib.metadata import version

    import airas_eval

    assert airas_eval.__version__ == version("airas-eval")

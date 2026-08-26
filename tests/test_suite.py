import importlib.util
import json

import numpy as np
import pytest

from airas_eval import evaluate
from airas_eval.suite import SUITES

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


def test_multiclass_full_inputs():
    report = evaluate(
        "classification",
        {
            "predicted_labels": Y_PRED_MC,
            "reference_labels": Y_TRUE_MC,
            "probabilities": PROBS_MC,
        },
    )
    for name in (
        "accuracy",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "recall_weighted",
        "balanced_accuracy",
        "matthews_corrcoef",
        "cohen_kappa",
        "log_loss",
        "expected_calibration_error",
        "top_5_accuracy",
    ):
        assert name in report.metrics, name
    # Binary-only metrics must be skipped with a reason, not silently absent.
    for name in ("auroc", "average_precision", "brier_score"):
        assert name in report.skipped
        assert "binary-only" in report.skipped[name]


def test_multiclass_without_probabilities_reports_skips():
    report = evaluate(
        "classification",
        {"predicted_labels": Y_PRED_MC, "reference_labels": Y_TRUE_MC},
    )
    assert "accuracy" in report.metrics
    assert report.skipped["log_loss"] == "requires input(s): probabilities"
    assert "top_5_accuracy" in report.skipped


def test_binary_computes_score_metrics():
    report = evaluate(
        "classification",
        {
            "predicted_labels": Y_PRED_BIN,
            "reference_labels": Y_TRUE_BIN,
            "probabilities": PROBS_BIN,
        },
    )
    assert 0.5 < report.metrics["auroc"] <= 1.0
    assert "average_precision" in report.metrics
    assert "brier_score" in report.metrics
    assert "top-5" in report.skipped["top_5_accuracy"]


def test_contract_is_fail_closed():
    with pytest.raises(KeyError):
        evaluate("no-such-task", {})
    with pytest.raises(ValueError, match="unknown input keys"):
        evaluate(
            "classification",
            {
                "predicted_labels": [0],
                "reference_labels": [0],
                "predicted_lables": [0],
            },
        )
    with pytest.raises(ValueError, match="missing required"):
        evaluate("classification", {"predicted_labels": [0, 1]})


def test_regression_suite_and_undefined_mape():
    report = evaluate(
        "regression",
        {"predicted_values": [1.0, 2.0, 3.5], "reference_values": [1.0, 2.0, 0.0]},
    )
    assert "rmse" in report.metrics
    assert "mape" in report.skipped
    assert "undefined" in report.skipped["mape"]


def test_retrieval_suite():
    report = evaluate(
        "retrieval",
        {
            "ranked_lists": [["a", "b", "c"], ["x", "y", "z"]],
            "relevant_sets": [["b"], ["x", "z"]],
        },
    )
    assert report.metrics["precision_at_1"] == pytest.approx(0.5)
    assert report.metrics["mean_reciprocal_rank"] == pytest.approx((0.5 + 1.0) / 2)
    assert "ndcg_at_10" in report.skipped


def test_text_generation_wrappers_present_or_skipped():
    report = evaluate(
        "text_generation",
        {
            "predicted_texts": ["the cat sat"],
            "reference_texts": ["the cat sat"],
        },
    )
    if importlib.util.find_spec("sacrebleu"):
        assert report.metrics["bleu"] == pytest.approx(100.0)
    else:
        assert "optional dependency" in report.skipped["bleu"]


def test_provenance_versions_and_input_hash():
    inputs = {"predicted_labels": [0, 1], "reference_labels": [0, 1]}
    a = evaluate("classification", inputs)
    b = evaluate("classification", inputs)
    c = evaluate(
        "classification", {"predicted_labels": [1, 1], "reference_labels": [0, 1]}
    )
    assert a.provenance["inputs_sha256"] == b.provenance["inputs_sha256"]
    assert a.provenance["inputs_sha256"] != c.provenance["inputs_sha256"]
    versions = a.provenance["versions"]
    assert "airas-eval" in versions
    assert versions["scikit-learn"] not in ("", "not installed")
    assert a.provenance["suite_signature"].startswith("classification/v1")


def test_report_round_trips_through_json():
    report = evaluate(
        "clustering",
        {"predicted_labels": [0, 0, 1, 1], "reference_labels": [1, 1, 0, 0]},
    )
    parsed = json.loads(report.to_json())
    assert parsed["metrics"]["adjusted_rand_index"] == pytest.approx(1.0)
    assert parsed["task_type"] == "clustering"


def test_every_suite_metric_name_is_unique():
    for task_type, suite in SUITES.items():
        names = [m.name for m in suite.metrics]
        assert len(names) == len(set(names)), task_type


def test_numpy_inputs_are_hashable():
    report = evaluate(
        "classification",
        {
            "predicted_labels": np.array([0, 1, 1]),
            "reference_labels": np.array([0, 1, 0]),
        },
    )
    assert len(report.provenance["inputs_sha256"]) == 64

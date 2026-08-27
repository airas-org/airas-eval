import json

import numpy as np
import pytest

from airas_eval import aggregate_reports, compare, evaluate, validate_inputs
from airas_eval.tasks import TASKS


def _skipped_names(report) -> set[str]:
    return {
        name
        for code, entries in report.skipped.items()
        for name in (entries if isinstance(entries, list) else entries.keys())
    }


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
    report = evaluate("classification", CLS_FULL)
    assert set(report.metrics) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "balanced_accuracy",
        "matthews_corrcoef",
        "log_loss",
        "expected_calibration_error",
        "top_5_accuracy",
    }
    assert report.inputs_summary == {
        "n_examples": N,
        "n_classes": 6,
    }
    assert not any(report.skipped.values())
    assert report.omitted_optional_inputs == []


def test_missing_optional_input_is_coded_and_surfaced():
    report = evaluate("classification", CLS_NO_PROBS)
    assert "accuracy" in report.metrics
    assert "log_loss" in report.skipped["missing_optional_input"]
    assert "n_classes" in report.skipped["missing_optional_input"]
    assert report.omitted_optional_inputs == ["probabilities"]


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_binary_classification_task():
    report = evaluate("binary_classification", BIN_FULL)
    assert 0.5 < report.metrics["auroc"] <= 1.0
    for name in ("precision", "recall", "f1", "brier_score", "average_precision"):
        assert f"{name}" in report.metrics
    assert not any(report.skipped.values())
    # a binary task fed one-class references: data problem, not a shape problem
    one_class = evaluate(
        "binary_classification",
        {
            "predicted_labels": [0, 1, 0],
            "reference_labels": [0, 0, 0],
            "probabilities": [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]],
        },
    )
    assert "auroc" in one_class.skipped["undefined_on_data"]


def test_multiclass_task_marks_top5_not_applicable_for_few_classes():
    report = evaluate("classification", BIN_FULL)
    assert "top_5_accuracy" in report.skipped["not_applicable"]


def test_contract_is_fail_closed():
    with pytest.raises(KeyError):
        evaluate("no-such-task", {})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {})  # required inputs missing
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {"main": CLS_NO_PROBS})  # no wrapping object
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {**CLS_NO_PROBS, "extra": {}})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "classification",
            {
                "predicted_labels": [0],
                "reference_labels": [0],
                "typo": [0],
            },
        )
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {"predicted_labels": [0, 1]})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "candidate_ranking",
            {
                "predicted_scores": [float("nan")],
                "reference_scores": [1.0],
            },
        )


def test_malformed_data_raises_instead_of_skipping():
    # A length mismatch is a contract violation, not "undefined on this
    # data" — the evaluator must not swallow it into `skipped`.
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate(
            "candidate_ranking",
            {
                "predicted_scores": [1.0, 2.0],
                "reference_scores": [1.0],
            },
        )


def test_search_task():
    report = evaluate("search", SEARCH_FULL)
    assert report.metrics["best_score"] == pytest.approx(93.1)
    assert report.metrics["final_regret"] == pytest.approx(1.27)
    assert report.metrics["evaluations_to_best"] == 4
    assert report.metrics["mean_evaluated_score"] == pytest.approx(90.575)
    assert report.curves["best_so_far"] == [88.0, 91.2, 91.2, 93.1]
    assert report.inputs_summary == {"n_evaluations": 4}
    without_oracle = evaluate("search", {"evaluated_scores": [88.0, 91.2]})
    assert "final_regret" in without_oracle.skipped["missing_optional_input"]
    assert without_oracle.omitted_optional_inputs == ["oracle_best"]
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        evaluate("search", {"evaluated_scores": [95.0], "oracle_best": 94.37})


def test_candidate_ranking_task():
    reference = [float(i) for i in range(40)]
    predicted = [v + (5.0 if i % 7 == 0 else 0.0) for i, v in enumerate(reference)]
    report = evaluate(
        "candidate_ranking",
        {
            "predicted_scores": predicted,
            "reference_scores": reference,
        },
    )
    assert 0.0 < report.metrics["kendall_tau"] <= 1.0
    assert "precision_at_top_10pct" in report.metrics
    assert "best_true_rank_in_top_10" in report.metrics
    assert report.inputs_summary == {"n_candidates": 40}
    small = evaluate("candidate_ranking", RANK_SMALL)
    assert "best_true_rank_in_top_10" in small.skipped["undefined_on_data"]
    assert "precision_at_top_10pct" in small.skipped["undefined_on_data"]


def test_multiobjective_task():
    report = evaluate("multiobjective", MOO)
    assert report.metrics["pareto_front_size"] == 2.0
    assert report.metrics["hypervolume_2d"] == pytest.approx(6.9)
    assert report.metrics["spacing"] == pytest.approx(0.0)  # 2-point front
    assert report.curves["pareto_front"] == [[0.1, 5.0], [0.2, 2.0]]
    assert report.inputs_summary == {
        "n_points": 3,
        "n_objectives": 2,
    }
    for name in ("igd", "gd"):
        assert f"{name}" in report.skipped["missing_optional_input"]
    assert report.omitted_optional_inputs == ["reference_front"]


def test_undefined_metric_is_skipped_with_code():
    report = evaluate(
        "candidate_ranking",
        {
            "predicted_scores": [1.0, 1.0, 1.0],
            "reference_scores": [1.0, 2.0, 3.0],
        },
    )
    assert "kendall_tau" in report.skipped["undefined_on_data"]


NAS_SEARCH_FULL = {
    **SEARCH_FULL,
    "evaluation_costs": [100.0, 200.0, 100.0, 300.0],
    "search_space_scores": [80.0, 88.0, 90.0, 91.2, 93.1, 94.37, 85.0, 70.0],
}


def test_nas_tasks_are_supersets_of_generic_tasks():
    pairs = (
        ("nas_pre_training", "search", SEARCH_FULL),
        ("nas_pre_training", "candidate_ranking", RANK_SMALL),
        ("nas_post_training", "classification", CLS_FULL),
    )
    for nas_task, generic_task, inputs in pairs:
        a = evaluate(nas_task, inputs)
        b = evaluate(generic_task, inputs)
        # same numbers for the shared metrics ...
        assert all(a.metrics[k] == v for k, v in b.metrics.items())
        # ... and the NAS extras exist in the report as coded skips, since the
        # generic inputs carry none of the NAS reference data
        declared_a = set(a.metrics) | _skipped_names(a)
        declared_b = set(b.metrics) | _skipped_names(b)
        assert declared_a > declared_b, nas_task
        assert a.provenance["task_signature"].startswith(f"{nas_task}/v1@")


def test_nas_pre_training_requires_search_or_predictor():
    with pytest.raises(ValueError, match="neither was given"):
        evaluate("nas_pre_training", {})
    with pytest.raises(ValueError, match="given together"):
        evaluate("nas_pre_training", {"predicted_scores": [1.0, 2.0]})
    report = evaluate("nas_pre_training", RANK_SMALL)
    assert "evaluated_scores" in report.omitted_optional_inputs
    assert "best_score" in report.skipped["missing_optional_input"]


def test_nas_pre_training_search_specific_metrics():
    report = evaluate("nas_pre_training", NAS_SEARCH_FULL)
    m = report.metrics
    assert m["cost_to_best"] == 700.0
    assert m["search_space_fraction_better"] == pytest.approx(1 / 8)
    assert m["gain_over_random_search"] > 0  # beat a 4-draw random baseline
    assert m["relative_improvement_over_random"] == pytest.approx(
        (93.1 - 86.45875) / 86.45875
    )
    assert report.curves["best_so_far_vs_cost"][-1] == [700.0, 93.1]
    assert report.inputs_summary["total_cost"] == 700.0
    assert report.inputs_summary["n_search_space"] == 8
    assert {"predicted_scores", "reference_scores"} <= set(
        report.omitted_optional_inputs
    )
    with pytest.raises(ValueError, match="one entry per evaluated score"):
        evaluate(
            "nas_pre_training",
            {**SEARCH_FULL, "evaluation_costs": [1.0]},
        )


def test_nas_post_training_architecture_baselines():
    inputs = {
        **CLS_NO_PROBS,
        "random_architecture_accuracies": [0.2, 0.3, 0.4],
        "oracle_test_best": 0.95,
    }
    report = evaluate("nas_post_training", inputs)
    acc = report.metrics["accuracy"]
    assert report.metrics["relative_improvement_over_random"] == pytest.approx(
        (acc - 0.3) / 0.3
    )
    assert report.metrics["fraction_of_random_better"] == 0.0
    assert report.metrics["test_regret"] == pytest.approx(0.95 - acc)
    assert report.inputs_summary["n_random_architectures"] == 3
    assert "points" in report.omitted_optional_inputs
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate(
            "nas_post_training",
            {
                **CLS_NO_PROBS,
                "random_architecture_accuracies": [70.0],
            },
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate(
            "nas_post_training",
            {**CLS_NO_PROBS, "oracle_test_best": 94.0},
        )
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        evaluate(
            "nas_post_training",
            {**CLS_NO_PROBS, "oracle_test_best": 0.0},
        )


def test_nas_post_training_tradeoff_metrics():
    report = evaluate(
        "nas_post_training",
        {
            **CLS_NO_PROBS,
            "points": [[0.1, 5.0], [0.2, 2.0], [0.3, 4.0]],
            "reference_point": [1.0, 10.0],
        },
    )
    assert report.metrics["hypervolume_2d"] == pytest.approx(6.9)
    assert report.curves["pareto_front"] == [[0.1, 5.0], [0.2, 2.0]]
    assert "igd" in report.skipped["missing_optional_input"]
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("nas_post_training", {"points": [[0.1, 1.0]]})  # labels missing


def test_nas_pre_training_predictor_top_fraction_correlations():
    ref = [float(i) for i in range(40)]
    pred = ref[:]
    pred[35], pred[39] = pred[39], pred[35]  # swap two of the true top-10%
    report = evaluate(
        "nas_pre_training",
        {"predicted_scores": pred, "reference_scores": ref},
    )
    assert report.metrics["kendall_tau"] > report.metrics["kendall_tau_top_10pct"]
    assert report.metrics["spearman_rho_top_10pct"] < 1.0


def test_provenance_signature_and_hashes():
    inputs = {"predicted_labels": [0, 1], "reference_labels": [0, 1]}
    a = evaluate("classification", inputs)
    b = evaluate("classification", inputs)
    c = evaluate(
        "classification",
        {"predicted_labels": [1, 1], "reference_labels": [0, 1]},
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
            "predicted_labels": np.array([0, 1, 1]),
            "reference_labels": np.array([0, 1, 0]),
        },
    )
    assert len(report.provenance["inputs_sha256"]) == 64


def test_every_task_reports_input_sizes_outside_metrics():
    for task_type, task in TASKS.items():
        summary = [b.name for b in task.summary]
        sizes = ("n_", "total_")
        assert summary and all(n.startswith(sizes) for n in summary), task_type
        assert not any(b.name.startswith(sizes) for b in task.metrics), task_type


def test_every_task_is_between_five_and_twentyfive_metrics():
    # curves are metrics too (non-scalar); summary entries are not
    for task_type, task in TASKS.items():
        n = len(task.metrics) + len(task.curves)
        assert 5 <= n <= 25, (task_type, n)


def test_version_comes_from_installed_metadata():
    from importlib.metadata import version

    import airas_eval

    assert airas_eval.__version__ == version("airas-eval")


def test_validate_inputs_without_scoring():
    assert validate_inputs("nas_pre_training", RANK_SMALL) == [
        "predicted_scores",
        "reference_scores",
    ]
    with pytest.raises(ValueError, match="invalid inputs"):
        validate_inputs("nas_pre_training", {**RANK_SMALL, "unknown": 1})


def test_input_schema_is_derived_from_the_models():
    assert TASKS["nas_pre_training"].input_constraints()
    schema = TASKS["nas_post_training"].input_schema()
    assert schema["required"] == ["predicted_labels", "reference_labels"]
    assert schema["additionalProperties"] is False
    assert {"oracle_test_best", "points", "probabilities"} <= set(schema["properties"])
    assert schema["properties"]["oracle_test_best"]["description"]
    assert schema["title"] == "nas_post_training inputs"
    pre = TASKS["nas_pre_training"].input_schema()
    assert "required" not in pre or pre["required"] == []
    assert {"required": ["evaluated_scores"]} in pre["anyOf"]
    assert pre["dependentRequired"]["predicted_scores"] == ["reference_scores"]


def _seed_reports(n: int) -> list[dict]:
    reports = []
    for seed in range(n):
        r = np.random.default_rng(seed)
        y_true = r.integers(0, 3, size=60).tolist()
        y_pred = np.where(
            r.random(60) < 0.8, y_true, r.integers(0, 3, size=60)
        ).tolist()
        report = evaluate(
            "classification",
            {
                "predicted_labels": y_pred,
                "reference_labels": y_true,
            },
        )
        reports.append(json.loads(report.to_json()))
    return reports


def test_aggregate_reports_over_seeds():
    agg = aggregate_reports(_seed_reports(3))
    assert agg.n_reports == 3
    assert agg.metrics["accuracy"]["n"] == 3.0
    assert 0.0 <= agg.metrics["accuracy"]["std"] < 0.2
    assert agg.inputs_summary["n_examples"]["mean"] == 60.0
    assert agg.incomplete == {}
    assert len(agg.provenance["inputs_sha256"]) == 3
    json.loads(agg.to_json())


def test_aggregate_rejects_mixed_signatures_and_reports_incomplete():
    a = _seed_reports(1)[0]
    b = json.loads(evaluate("search", {"evaluated_scores": [1.0, 2.0]}).to_json())
    with pytest.raises(ValueError, match="task signatures"):
        aggregate_reports([a, b])
    c = json.loads(
        evaluate(
            "classification",
            {**CLS_NO_PROBS, "probabilities": None},
        ).to_json()
    )
    d = json.loads(evaluate("classification", CLS_FULL).to_json())
    agg = aggregate_reports([c, d])
    assert agg.incomplete["log_loss"] == 1  # only the run with probabilities


def test_compare_paired_on_classification():
    r = np.random.default_rng(1)
    y_true = r.integers(0, 2, size=120).tolist()
    good = np.where(r.random(120) < 0.9, y_true, 1 - np.array(y_true)).tolist()
    bad = np.where(r.random(120) < 0.55, y_true, 1 - np.array(y_true)).tolist()
    result = compare(
        "nas_post_training",
        {"predicted_labels": good, "reference_labels": y_true},
        {"predicted_labels": bad, "reference_labels": y_true},
    )
    c = result.comparisons["correct"]
    assert c["mean_a"] > c["mean_b"]
    assert c["mean_diff"] == pytest.approx(c["mean_a"] - c["mean_b"])
    assert 0.0 < c["p_value"] < 0.05
    assert c["n_examples"] == 120.0
    json.loads(result.to_json())


def test_compare_requires_identical_reference_data():
    with pytest.raises(ValueError, match="identical reference"):
        compare(
            "classification",
            {
                "predicted_labels": [0, 1],
                "reference_labels": [0, 1],
            },
            {
                "predicted_labels": [0, 1],
                "reference_labels": [1, 1],
            },
        )


def test_compare_unavailable_without_per_example_scores():
    with pytest.raises(ValueError, match="no per-example"):
        compare("search", {"evaluated_scores": [1.0]}, {"evaluated_scores": [2.0]})

import json

import numpy as np
import pytest

from airas_eval import aggregate_reports, compare, evaluate, validate_inputs
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
    report = evaluate("classification", {"classification": CLS_FULL})
    assert set(report.metrics) == {
        "classification.accuracy",
        "classification.precision_macro",
        "classification.recall_macro",
        "classification.f1_macro",
        "classification.balanced_accuracy",
        "classification.matthews_corrcoef",
        "classification.log_loss",
        "classification.expected_calibration_error",
        "classification.top_5_accuracy",
    }
    assert report.inputs_summary == {
        "classification.n_examples": N,
        "classification.n_classes": 6,
    }
    assert report.skipped == {}
    assert report.omitted_optional_inputs == []


def test_missing_optional_input_is_coded_and_surfaced():
    report = evaluate("classification", {"classification": CLS_NO_PROBS})
    assert "classification.accuracy" in report.metrics
    assert report.skipped["classification.log_loss"]["code"] == "missing_optional_input"
    assert (
        report.skipped["classification.n_classes"]["code"] == "missing_optional_input"
    )
    assert report.omitted_optional_inputs == ["classification.probabilities"]


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_binary_classification_task():
    report = evaluate("binary_classification", {"binary_classification": BIN_FULL})
    assert 0.5 < report.metrics["binary_classification.auroc"] <= 1.0
    for name in ("precision", "recall", "f1", "brier_score", "average_precision"):
        assert f"binary_classification.{name}" in report.metrics
    assert report.skipped == {}
    # a binary task fed one-class references: data problem, not a shape problem
    one_class = evaluate(
        "binary_classification",
        {
            "binary_classification": {
                "predicted_labels": [0, 1, 0],
                "reference_labels": [0, 0, 0],
                "probabilities": [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]],
            }
        },
    )
    assert (
        one_class.skipped["binary_classification.auroc"]["code"] == "undefined_on_data"
    )


def test_multiclass_task_marks_top5_not_applicable_for_few_classes():
    report = evaluate("classification", {"classification": BIN_FULL})
    assert report.skipped["classification.top_5_accuracy"]["code"] == "not_applicable"


def test_contract_is_fail_closed():
    with pytest.raises(KeyError):
        evaluate("no-such-task", {})
    with pytest.raises(ValueError, match="unknown group"):
        evaluate("classification", CLS_NO_PROBS)  # ungrouped
    with pytest.raises(ValueError, match="required group"):
        evaluate("classification", {})
    with pytest.raises(ValueError, match="unknown group"):
        evaluate("classification", {"classification": CLS_NO_PROBS, "extra": {}})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "classification",
            {
                "classification": {
                    "predicted_labels": [0],
                    "reference_labels": [0],
                    "typo": [0],
                }
            },
        )
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate("classification", {"classification": {"predicted_labels": [0, 1]}})
    with pytest.raises(ValueError, match="invalid inputs"):
        evaluate(
            "candidate_ranking",
            {
                "classification": {
                    "predicted_scores": [float("nan")],
                    "reference_scores": [1.0],
                }
            },
        )


def test_malformed_data_raises_instead_of_skipping():
    # A length mismatch is a contract violation, not "undefined on this
    # data" — the evaluator must not swallow it into `skipped`.
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate(
            "candidate_ranking",
            {
                "candidate_ranking": {
                    "predicted_scores": [1.0, 2.0],
                    "reference_scores": [1.0],
                }
            },
        )


def test_search_task():
    report = evaluate("search", {"search": SEARCH_FULL})
    assert report.metrics["search.best_score"] == pytest.approx(93.1)
    assert report.metrics["search.final_regret"] == pytest.approx(1.27)
    assert report.metrics["search.evaluations_to_best"] == 4
    assert report.metrics["search.mean_evaluated_score"] == pytest.approx(90.575)
    assert report.curves["search.best_so_far"] == [88.0, 91.2, 91.2, 93.1]
    assert report.inputs_summary == {"search.n_evaluations": 4}
    without_oracle = evaluate("search", {"search": {"evaluated_scores": [88.0, 91.2]}})
    assert (
        without_oracle.skipped["search.final_regret"]["code"]
        == "missing_optional_input"
    )
    assert without_oracle.omitted_optional_inputs == ["search.oracle_best"]
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        evaluate(
            "search", {"search": {"evaluated_scores": [95.0], "oracle_best": 94.37}}
        )


def test_candidate_ranking_task():
    reference = [float(i) for i in range(40)]
    predicted = [v + (5.0 if i % 7 == 0 else 0.0) for i, v in enumerate(reference)]
    report = evaluate(
        "candidate_ranking",
        {
            "candidate_ranking": {
                "predicted_scores": predicted,
                "reference_scores": reference,
            }
        },
    )
    assert 0.0 < report.metrics["candidate_ranking.kendall_tau"] <= 1.0
    assert "candidate_ranking.precision_at_top_10pct" in report.metrics
    assert "candidate_ranking.best_true_rank_in_top_10" in report.metrics
    assert report.inputs_summary == {"candidate_ranking.n_candidates": 40}
    small = evaluate("candidate_ranking", {"candidate_ranking": RANK_SMALL})
    assert (
        small.skipped["candidate_ranking.best_true_rank_in_top_10"]["code"]
        == "undefined_on_data"
    )
    assert (
        small.skipped["candidate_ranking.precision_at_top_10pct"]["code"]
        == "undefined_on_data"
    )


def test_multiobjective_task():
    report = evaluate("multiobjective", {"multiobjective": MOO})
    assert report.metrics["multiobjective.pareto_front_size"] == 2.0
    assert report.metrics["multiobjective.hypervolume_2d"] == pytest.approx(6.9)
    assert report.metrics["multiobjective.spacing"] == pytest.approx(
        0.0
    )  # 2-point front
    assert report.curves["multiobjective.pareto_front"] == [[0.1, 5.0], [0.2, 2.0]]
    assert report.inputs_summary == {
        "multiobjective.n_points": 3,
        "multiobjective.n_objectives": 2,
    }
    for name in ("igd", "gd"):
        assert (
            report.skipped[f"multiobjective.{name}"]["code"] == "missing_optional_input"
        )
    assert report.omitted_optional_inputs == ["multiobjective.reference_front"]


def test_undefined_metric_is_skipped_with_code():
    report = evaluate(
        "candidate_ranking",
        {
            "candidate_ranking": {
                "predicted_scores": [1.0, 1.0, 1.0],
                "reference_scores": [1.0, 2.0, 3.0],
            }
        },
    )
    assert (
        report.skipped["candidate_ranking.kendall_tau"]["code"] == "undefined_on_data"
    )


NAS_SEARCH_FULL = {
    **SEARCH_FULL,
    "evaluation_costs": [100.0, 200.0, 100.0, 300.0],
    "search_space_scores": [80.0, 88.0, 90.0, 91.2, 93.1, 94.37, 85.0, 70.0],
}


def test_nas_tasks_are_supersets_of_generic_tasks():
    pairs = {
        ("nas_pre_training", "search"): ("search", SEARCH_FULL),
        ("nas_pre_training", "predictor"): ("candidate_ranking", RANK_SMALL),
        ("nas_post_training", "architecture"): ("classification", CLS_FULL),
    }
    for (nas_task, group), (generic_task, inputs) in pairs.items():
        a = evaluate(nas_task, {group: inputs})
        b = evaluate(generic_task, {generic_task: inputs})
        prefix = len(generic_task) + 1
        # same numbers for the shared metrics ...
        assert all(
            a.metrics[f"{group}.{k[prefix:]}"] == v for k, v in b.metrics.items()
        )
        # ... and the NAS extras exist in the report as coded skips, since the
        # generic inputs carry none of the NAS reference data
        declared_a = {k for k in set(a.metrics) | set(a.skipped) if k.startswith(group)}
        declared_b = set(b.metrics) | set(b.skipped)
        assert len(declared_a) > len(declared_b), nas_task
        assert a.provenance["task_signature"].startswith(f"{nas_task}/v1@")


def test_nas_pre_training_requires_at_least_one_group():
    with pytest.raises(ValueError, match="at least one of the groups"):
        evaluate("nas_pre_training", {})
    report = evaluate("nas_pre_training", {"predictor": RANK_SMALL})
    assert "search" in report.omitted_optional_inputs
    assert report.skipped["search.best_score"]["code"] == "missing_optional_input"


def test_nas_pre_training_search_specific_metrics():
    report = evaluate("nas_pre_training", {"search": NAS_SEARCH_FULL})
    m = report.metrics
    assert m["search.cost_to_best"] == 700.0
    assert m["search.search_space_fraction_better"] == pytest.approx(1 / 8)
    assert m["search.gain_over_random_search"] > 0  # beat a 4-draw random baseline
    assert m["search.relative_improvement_over_random"] == pytest.approx(
        (93.1 - 86.45875) / 86.45875
    )
    assert report.curves["search.best_so_far_vs_cost"][-1] == [700.0, 93.1]
    assert report.inputs_summary["search.total_cost"] == 700.0
    assert report.inputs_summary["search.n_search_space"] == 8
    assert report.omitted_optional_inputs == ["predictor"]
    with pytest.raises(ValueError, match="one entry per evaluated score"):
        evaluate(
            "nas_pre_training",
            {"search": {**SEARCH_FULL, "evaluation_costs": [1.0]}},
        )


def test_nas_post_training_architecture_baselines():
    inputs = {
        **CLS_NO_PROBS,
        "random_architecture_accuracies": [0.2, 0.3, 0.4],
        "oracle_test_best": 0.95,
    }
    report = evaluate("nas_post_training", {"architecture": inputs})
    acc = report.metrics["architecture.accuracy"]
    assert report.metrics[
        "architecture.relative_improvement_over_random"
    ] == pytest.approx((acc - 0.3) / 0.3)
    assert report.metrics["architecture.fraction_of_random_better"] == 0.0
    assert report.metrics["architecture.test_regret"] == pytest.approx(0.95 - acc)
    assert report.inputs_summary["architecture.n_random_architectures"] == 3
    assert "tradeoff" in report.omitted_optional_inputs
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate(
            "nas_post_training",
            {
                "architecture": {
                    **CLS_NO_PROBS,
                    "random_architecture_accuracies": [70.0],
                }
            },
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate(
            "nas_post_training",
            {"architecture": {**CLS_NO_PROBS, "oracle_test_best": 94.0}},
        )
    with pytest.raises(ValueError, match="exceeds the benchmark optimum"):
        evaluate(
            "nas_post_training",
            {"architecture": {**CLS_NO_PROBS, "oracle_test_best": 0.0}},
        )


def test_nas_post_training_tradeoff_group():
    report = evaluate(
        "nas_post_training",
        {
            "architecture": CLS_NO_PROBS,
            "tradeoff": {
                "points": [[0.1, 5.0], [0.2, 2.0], [0.3, 4.0]],
                "reference_point": [1.0, 10.0],
            },
        },
    )
    assert report.metrics["tradeoff.hypervolume_2d"] == pytest.approx(6.9)
    assert report.curves["tradeoff.pareto_front"] == [[0.1, 5.0], [0.2, 2.0]]
    with pytest.raises(ValueError, match="required group"):
        evaluate("nas_post_training", {"tradeoff": {"points": [[0.1, 1.0]]}})


def test_nas_pre_training_predictor_top_fraction_correlations():
    ref = [float(i) for i in range(40)]
    pred = ref[:]
    pred[35], pred[39] = pred[39], pred[35]  # swap two of the true top-10%
    report = evaluate(
        "nas_pre_training",
        {"predictor": {"predicted_scores": pred, "reference_scores": ref}},
    )
    assert (
        report.metrics["predictor.kendall_tau"]
        > report.metrics["predictor.kendall_tau_top_10pct"]
    )
    assert report.metrics["predictor.spearman_rho_top_10pct"] < 1.0


def test_provenance_signature_and_hashes():
    inputs = {
        "classification": {"predicted_labels": [0, 1], "reference_labels": [0, 1]}
    }
    a = evaluate("classification", inputs)
    b = evaluate("classification", inputs)
    c = evaluate(
        "classification",
        {"classification": {"predicted_labels": [1, 1], "reference_labels": [0, 1]}},
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
            "classification": {
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
            sizes = ("n_", "total_")
            assert summary and all(n.startswith(sizes) for n in summary), task_type
            assert not any(b.name.startswith(sizes) for b in group.bundle.metrics), (
                task_type
            )


def test_every_group_is_between_five_and_twelve_metrics():
    # curves are metrics too (non-scalar); summary entries are not
    for task_type, task in TASKS.items():
        for group in task.groups:
            n = len(group.bundle.metrics) + len(group.bundle.curves)
            assert 5 <= n <= 12, (task_type, group.name, n)


def test_version_comes_from_installed_metadata():
    from importlib.metadata import version

    import airas_eval

    assert airas_eval.__version__ == version("airas-eval")


def test_validate_inputs_without_scoring():
    assert validate_inputs("nas_pre_training", {"predictor": RANK_SMALL}) == [
        "predictor"
    ]
    with pytest.raises(ValueError, match="unknown group"):
        validate_inputs("nas_pre_training", {"unknown": RANK_SMALL})


def test_input_schema_is_derived_from_the_models():
    schema = TASKS["nas_post_training"].input_schema()
    assert schema["required"] == ["architecture"]
    assert set(schema["properties"]) == {"architecture", "tradeoff"}
    arch = schema["properties"]["architecture"]
    assert arch["additionalProperties"] is False
    assert set(arch["required"]) == {"predicted_labels", "reference_labels"}
    assert "oracle_test_best" in arch["properties"]
    assert arch["properties"]["oracle_test_best"]["description"]
    assert TASKS["nas_pre_training"].input_schema()["required"] == []
    assert TASKS["nas_pre_training"].input_schema()["minProperties"] == 1


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
                "classification": {
                    "predicted_labels": y_pred,
                    "reference_labels": y_true,
                }
            },
        )
        reports.append(json.loads(report.to_json()))
    return reports


def test_aggregate_reports_over_seeds():
    agg = aggregate_reports(_seed_reports(3))
    assert agg.n_reports == 3
    assert agg.metrics["classification.accuracy"]["n"] == 3.0
    assert 0.0 <= agg.metrics["classification.accuracy"]["std"] < 0.2
    assert agg.inputs_summary["classification.n_examples"]["mean"] == 60.0
    assert agg.incomplete == {}
    assert len(agg.provenance["inputs_sha256"]) == 3
    json.loads(agg.to_json())


def test_aggregate_rejects_mixed_signatures_and_reports_incomplete():
    a = _seed_reports(1)[0]
    b = json.loads(
        evaluate("search", {"search": {"evaluated_scores": [1.0, 2.0]}}).to_json()
    )
    with pytest.raises(ValueError, match="task signatures"):
        aggregate_reports([a, b])
    c = json.loads(
        evaluate(
            "classification",
            {"classification": {**CLS_NO_PROBS, "probabilities": None}},
        ).to_json()
    )
    d = json.loads(evaluate("classification", {"classification": CLS_FULL}).to_json())
    agg = aggregate_reports([c, d])
    assert (
        agg.incomplete["classification.log_loss"] == 1
    )  # only the run with probabilities


def test_compare_paired_on_classification():
    r = np.random.default_rng(1)
    y_true = r.integers(0, 2, size=120).tolist()
    good = np.where(r.random(120) < 0.9, y_true, 1 - np.array(y_true)).tolist()
    bad = np.where(r.random(120) < 0.55, y_true, 1 - np.array(y_true)).tolist()
    result = compare(
        "nas_post_training",
        {"architecture": {"predicted_labels": good, "reference_labels": y_true}},
        {"architecture": {"predicted_labels": bad, "reference_labels": y_true}},
    )
    c = result.comparisons["architecture.correct"]
    assert c["mean_a"] > c["mean_b"]
    assert c["mean_diff"] == pytest.approx(c["mean_a"] - c["mean_b"])
    assert 0.0 < c["p_value"] < 0.05
    assert c["n_examples"] == 120.0
    assert "no per-example" in result.skipped["tradeoff"]
    json.loads(result.to_json())


def test_compare_requires_identical_reference_data():
    with pytest.raises(ValueError, match="identical reference"):
        compare(
            "classification",
            {
                "classification": {
                    "predicted_labels": [0, 1],
                    "reference_labels": [0, 1],
                }
            },
            {
                "classification": {
                    "predicted_labels": [0, 1],
                    "reference_labels": [1, 1],
                }
            },
        )


def test_compare_skips_groups_without_per_example_scores():
    result = compare(
        "search",
        {"search": {"evaluated_scores": [1.0]}},
        {"search": {"evaluated_scores": [2.0]}},
    )
    assert result.comparisons == {}
    assert "no per-example" in result.skipped["search"]

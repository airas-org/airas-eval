"""Metric registry: which standard metrics exist, per task type.

The registry is the contract surface for AIRAS: an experimental design that
declares its metrics must name entries from this registry (or extend it via a
reviewed pull request — never via ad-hoc agent-written implementations).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    name: str
    module: str
    function: str
    pure: bool = True
    wrapped_package: str | None = None
    notes: str = ""


_M = "airas_eval.metrics"

REGISTRY: dict[str, list[MetricSpec]] = {
    "classification": [
        MetricSpec("accuracy", f"{_M}.classification", "accuracy"),
        MetricSpec("error_rate", f"{_M}.classification", "error_rate"),
        MetricSpec("top_k_accuracy", f"{_M}.classification", "top_k_accuracy"),
        MetricSpec("precision", f"{_M}.classification", "precision"),
        MetricSpec("recall", f"{_M}.classification", "recall"),
        MetricSpec("f1", f"{_M}.classification", "f1"),
        MetricSpec("balanced_accuracy", f"{_M}.classification", "balanced_accuracy"),
        MetricSpec("matthews_corrcoef", f"{_M}.classification", "matthews_corrcoef"),
        MetricSpec("cohen_kappa", f"{_M}.classification", "cohen_kappa"),
        MetricSpec("auroc", f"{_M}.classification", "auroc"),
        MetricSpec("average_precision", f"{_M}.classification", "average_precision"),
        MetricSpec("log_loss", f"{_M}.classification", "log_loss"),
        MetricSpec("brier_score", f"{_M}.classification", "brier_score"),
        MetricSpec(
            "expected_calibration_error",
            f"{_M}.classification",
            "expected_calibration_error",
            notes="binning variant: equal-width, argmax confidence",
        ),
    ],
    "regression": [
        MetricSpec("mse", f"{_M}.regression", "mse"),
        MetricSpec("rmse", f"{_M}.regression", "rmse"),
        MetricSpec("mae", f"{_M}.regression", "mae"),
        MetricSpec("mape", f"{_M}.regression", "mape"),
        MetricSpec("smape", f"{_M}.regression", "smape"),
        MetricSpec("r2_score", f"{_M}.regression", "r2_score"),
        MetricSpec("explained_variance", f"{_M}.regression", "explained_variance"),
        MetricSpec("pearson_r", f"{_M}.regression", "pearson_r"),
        MetricSpec("spearman_rho", f"{_M}.regression", "spearman_rho"),
        MetricSpec("kendall_tau", f"{_M}.regression", "kendall_tau"),
    ],
    "ranking": [
        MetricSpec("precision_at_k", f"{_M}.ranking", "precision_at_k"),
        MetricSpec("recall_at_k", f"{_M}.ranking", "recall_at_k"),
        MetricSpec("hit_rate_at_k", f"{_M}.ranking", "hit_rate_at_k"),
        MetricSpec("mean_reciprocal_rank", f"{_M}.ranking", "mean_reciprocal_rank"),
        MetricSpec("mean_average_precision", f"{_M}.ranking", "mean_average_precision"),
        MetricSpec("ndcg_at_k", f"{_M}.ranking", "ndcg_at_k"),
    ],
    "clustering": [
        MetricSpec("adjusted_rand_index", f"{_M}.clustering", "adjusted_rand_index"),
        MetricSpec(
            "normalized_mutual_info",
            f"{_M}.clustering",
            "normalized_mutual_info",
            notes="normalization variant: arithmetic mean (sklearn default)",
        ),
        MetricSpec("adjusted_mutual_info", f"{_M}.clustering", "adjusted_mutual_info"),
        MetricSpec("v_measure", f"{_M}.clustering", "v_measure"),
    ],
    "vision": [
        MetricSpec("pixel_accuracy", f"{_M}.vision", "pixel_accuracy"),
        MetricSpec("binary_iou", f"{_M}.vision", "binary_iou"),
        MetricSpec("dice_coefficient", f"{_M}.vision", "dice_coefficient"),
        MetricSpec("mean_iou", f"{_M}.vision", "mean_iou"),
        MetricSpec("psnr", f"{_M}.vision", "psnr"),
    ],
    "nlp": [
        MetricSpec("exact_match", f"{_M}.nlp", "exact_match"),
        MetricSpec("token_f1", f"{_M}.nlp", "token_f1"),
        MetricSpec("bleu", f"{_M}.nlp", "bleu", wrapped_package="sacrebleu"),
        MetricSpec("chrf", f"{_M}.nlp", "chrf", wrapped_package="sacrebleu"),
        MetricSpec("rouge_l", f"{_M}.nlp", "rouge_l", wrapped_package="rouge-score"),
    ],
    "structure": [
        MetricSpec("rmsd", f"{_M}.structure", "rmsd"),
        MetricSpec(
            "tm_score", f"{_M}.structure", "tm_score", wrapped_package="tmtools"
        ),
        MetricSpec("dockq", f"{_M}.structure", "dockq", wrapped_package="DockQ"),
        # lDDT / lDDT-PLI: reference implementation is OpenStructure, which is
        # not pip-installable (conda/container only). Planned as a wrapper; an
        # in-house lDDT would be the exact failure mode this library prevents.
    ],
    "complexity": [
        MetricSpec(
            "parameter_count",
            f"{_M}.complexity",
            "parameter_count",
            pure=False,
            wrapped_package="torch",
            notes="takes a model, not (pred, ref)",
        ),
        MetricSpec(
            "macs",
            f"{_M}.complexity",
            "macs",
            pure=False,
            wrapped_package="fvcore",
            notes="returns MACs with explicit counter/convention metadata",
        ),
    ],
    "stats": [
        MetricSpec("mean_std", f"{_M}.stats", "mean_std"),
        MetricSpec("bootstrap_ci", f"{_M}.stats", "bootstrap_ci"),
        MetricSpec("paired_permutation_test", f"{_M}.stats", "paired_permutation_test"),
    ],
}


@dataclass(frozen=True)
class ResolvedMetric:
    spec: MetricSpec
    fn: object = field(repr=False)


def metric_names(task_type: str) -> list[str]:
    if task_type not in REGISTRY:
        raise KeyError(f"unknown task type {task_type!r}; known: {sorted(REGISTRY)}")
    return [spec.name for spec in REGISTRY[task_type]]


def resolve(task_type: str, name: str) -> ResolvedMetric:
    """Import and return the metric function registered under (task_type, name)."""
    import importlib

    for spec in REGISTRY.get(task_type, []):
        if spec.name == name:
            module = importlib.import_module(spec.module)
            return ResolvedMetric(spec=spec, fn=getattr(module, spec.function))
    raise KeyError(f"no metric {name!r} registered for task type {task_type!r}")

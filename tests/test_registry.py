import pytest

from airas_eval.registry import REGISTRY, metric_names, resolve


def test_all_core_metrics_resolve():
    for task_type, specs in REGISTRY.items():
        for spec in specs:
            if spec.wrapped_package is not None:
                continue  # optional extras may be absent in the test env
            resolved = resolve(task_type, spec.name)
            assert callable(resolved.fn), f"{task_type}/{spec.name} is not callable"


def test_names_unique_within_task():
    for task_type, specs in REGISTRY.items():
        names = [s.name for s in specs]
        assert len(names) == len(set(names)), f"duplicate metric names in {task_type}"


def test_metric_names_and_unknown_task():
    assert "accuracy" in metric_names("classification")
    with pytest.raises(KeyError):
        metric_names("no-such-task")
    with pytest.raises(KeyError):
        resolve("classification", "no-such-metric")

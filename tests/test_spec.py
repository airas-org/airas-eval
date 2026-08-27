import pytest

from airas_eval.metrics import regression as _reg
from airas_eval.spec import Bundle, Group, MetricBinding, TaskSpec
from airas_eval.tasks import TASKS, _bundles
from airas_eval.tasks._inputs import CandidateRankingInputs

_PAIR = ("predicted_scores", "reference_scores")


def _bundle(metrics: tuple[MetricBinding, ...]) -> Bundle:
    return Bundle(input_model=CandidateRankingInputs, metrics=metrics)


def _task(metrics: tuple[MetricBinding, ...]) -> TaskSpec:
    return TaskSpec(task_type="test_task", groups=(Group("main", _bundle(metrics)),))


def test_signature_is_derived_and_stable():
    a = _task((MetricBinding("mse", _reg.mse, _PAIR, {}, description="t"),))
    b = _task((MetricBinding("mse", _reg.mse, _PAIR, {}, description="t"),))
    assert a.signature() == b.signature()
    assert a.signature().startswith("test_task/v1@")


def test_signature_changes_with_kwargs_bindings_and_groups():
    base = _task((MetricBinding("m", _reg.mape, _PAIR, {}, description="t"),))
    other_fn = _task((MetricBinding("m", _reg.smape, _PAIR, {}, description="t"),))
    with_kwargs = _task(
        (MetricBinding("m", _reg.mape, _PAIR, {"variant": 2}, description="t"),)
    )
    assert base.signature() != other_fn.signature()
    assert base.signature() != with_kwargs.signature()
    bundle = _bundle((MetricBinding("m", _reg.mape, _PAIR, {}, description="t"),))
    renamed = TaskSpec("test_task", (Group("other", bundle),))
    optional = TaskSpec(
        "test_task", (Group("main", bundle), Group("extra", bundle, required=False))
    )
    assert base.signature() != renamed.signature()
    assert base.signature() != optional.signature()


def test_bindings_need_a_description():
    with pytest.raises(ValueError, match="needs a description"):
        _bundle((MetricBinding("m", _reg.mse, _PAIR),))


def test_lambdas_are_rejected():
    with pytest.raises(ValueError, match="lambda"):
        _bundle((MetricBinding("m", lambda a, b: 0.0, _PAIR, {}, description="t"),))


def test_unknown_binding_inputs_are_rejected():
    with pytest.raises(ValueError, match="not present"):
        _bundle(
            (
                MetricBinding(
                    "m",
                    _reg.mse,
                    ("no_such_field", "also_missing"),
                    {},
                    description="t",
                ),
            )
        )


def test_duplicate_metric_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _bundle(
            (
                MetricBinding("m", _reg.mse, _PAIR, {}, description="t"),
                MetricBinding("m", _reg.mae, _PAIR, {}, description="t"),
            )
        )


def test_task_declaration_is_validated():
    bundle = _bundles.CANDIDATE_RANKING
    with pytest.raises(ValueError, match="no groups"):
        TaskSpec("t", ())
    with pytest.raises(ValueError, match="duplicate group"):
        TaskSpec("t", (Group("a", bundle), Group("a", bundle)))
    with pytest.raises(ValueError, match="identifier"):
        TaskSpec("t", (Group("bad-name", bundle),))
    # all-optional groups are allowed; the evaluator requires at least one
    TaskSpec("t", (Group("a", bundle, required=False),))


def test_registered_tasks_have_distinct_signatures():
    signatures = [task.signature() for task in TASKS.values()]
    assert len(signatures) == len(set(signatures))


def test_nas_bundles_extend_the_core_bindings_verbatim():
    pairs = {
        ("nas_pre_training", "search"): "search",
        ("nas_pre_training", "predictor"): "candidate_ranking",
        ("nas_post_training", "architecture"): "classification",
    }
    for (nas_task, group), generic_task in pairs.items():
        nas = TASKS[nas_task].group(group).bundle
        core = TASKS[generic_task].group(generic_task).bundle
        assert nas.metrics[: len(core.metrics)] == core.metrics
        assert nas.curves[: len(core.curves)] == core.curves
        assert nas.summary[: len(core.summary)] == core.summary
        assert len(nas.metrics) > len(core.metrics)
        # the NAS input model accepts everything the core one does
        assert set(core.input_model.model_fields) <= set(nas.input_model.model_fields)


def test_description_does_not_affect_signature():
    bundle = _bundles.CANDIDATE_RANKING
    a = TaskSpec("t", (Group("main", bundle),), description="one")
    b = TaskSpec("t", (Group("main", bundle),), description="two")
    assert a.signature() == b.signature()


def test_every_metric_parameter_is_pinned_explicitly():
    """A default value on a metric function is not part of the signature, so
    bundles must pin every parameter beyond the bound inputs."""
    import inspect

    for task in TASKS.values():
        for group in task.groups:
            for b in group.bundle.bindings:
                params = list(inspect.signature(b.fn).parameters.values())
                extra = [p.name for p in params[len(b.inputs) :]]
                assert set(extra) <= set(b.kwargs), (task.task_type, b.name, extra)
                assert set(b.kwargs) <= set(extra), (task.task_type, b.name)


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        TASKS["classification"] = None  # type: ignore[index]

import inspect

import pytest

from airas_eval.metrics import regression as _reg
from airas_eval.spec import MetricBinding, MetricSet, TaskSpec
from airas_eval.tasks import TASKS
from airas_eval.tasks.generic import _metric_sets
from airas_eval.tasks.generic._inputs import CandidateRankingInputs

_PAIR = ("predicted_scores", "reference_scores")


def _task(metrics: tuple[MetricBinding, ...], task_type: str = "test_task") -> TaskSpec:
    return TaskSpec(
        task_type=task_type, input_model=CandidateRankingInputs, metrics=metrics
    )


def _b(name: str, fn, inputs=_PAIR, kwargs=None) -> MetricBinding:
    return MetricBinding(name, fn, inputs, kwargs or {}, description="t")


def test_signature_is_derived_and_stable():
    a = _task((_b("mse", _reg.mse),))
    b = _task((_b("mse", _reg.mse),))
    assert a.signature() == b.signature()
    assert a.signature().startswith("test_task/v1@")


def test_signature_changes_with_kwargs_bindings_and_inputs():
    base = _task((_b("m", _reg.mape),))
    other_fn = _task((_b("m", _reg.smape),))
    with_kwargs = _task((_b("m", _reg.mape, kwargs={"variant": 2}),))
    assert base.signature() != other_fn.signature()
    assert base.signature() != with_kwargs.signature()
    assert base.signature() != _task((_b("m", _reg.mape),), "other").signature()


def test_bindings_need_a_description():
    with pytest.raises(ValueError, match="needs a description"):
        _task((MetricBinding("m", _reg.mse, _PAIR),))


def test_lambdas_are_rejected():
    with pytest.raises(ValueError, match="lambda"):
        _task((_b("m", lambda a, b: 0.0),))


def test_unknown_binding_inputs_are_rejected():
    with pytest.raises(ValueError, match="not present"):
        _task((_b("m", _reg.mse, ("no_such_field", "also_missing")),))


def test_duplicate_metric_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _task((_b("m", _reg.mse), _b("m", _reg.mae)))


def test_task_declaration_is_validated():
    with pytest.raises(ValueError, match="no metrics"):
        TaskSpec("t", CandidateRankingInputs, ())
    with pytest.raises(ValueError, match="identifier"):
        _task((_b("m", _reg.mse),), "bad-name")
    with pytest.raises(ValueError, match="at least one metric set"):
        TaskSpec.from_sets("t", CandidateRankingInputs)


def test_from_sets_concatenates_in_order():
    first = MetricSet(metrics=(_b("a", _reg.mse),), notes="one")
    second = MetricSet(
        metrics=(_b("b", _reg.mae),), notes="two", provenance_packages=("scipy",)
    )
    task = TaskSpec.from_sets("t", CandidateRankingInputs, first, second)
    assert [b.name for b in task.metrics] == ["a", "b"]
    assert task.notes == "one。two"
    assert task.provenance_packages == ("numpy", "scipy")


def test_registered_tasks_have_distinct_signatures():
    signatures = [task.signature() for task in TASKS.values()]
    assert len(signatures) == len(set(signatures))


def test_nas_tasks_extend_the_core_bindings_verbatim():
    pre, post = TASKS["nas_pre_training"], TASKS["nas_post_training"]
    for nas, core in (
        (pre, TASKS["search"]),
        (pre, TASKS["candidate_ranking"]),
        (post, TASKS["classification"]),
        (post, TASKS["multiobjective"]),
    ):
        assert all(b in nas.metrics for b in core.metrics)
        assert all(b in nas.curves for b in core.curves)
        assert all(b in nas.summary for b in core.summary)
        # the NAS input model accepts everything the core one does
        assert set(core.input_model.model_fields) <= set(nas.input_model.model_fields)
    assert len(pre.metrics) > len(TASKS["search"].metrics) + len(
        TASKS["candidate_ranking"].metrics
    )


def test_description_does_not_affect_signature():
    a = TaskSpec.from_sets(
        "t", CandidateRankingInputs, _metric_sets.CANDIDATE_RANKING, description="one"
    )
    b = TaskSpec.from_sets(
        "t", CandidateRankingInputs, _metric_sets.CANDIDATE_RANKING, description="two"
    )
    assert a.signature() == b.signature()


def test_every_metric_parameter_is_pinned_explicitly():
    """A default value on a metric function is not part of the signature, so
    tasks must pin every parameter beyond the bound inputs."""
    for task in TASKS.values():
        for b in task.bindings:
            params = list(inspect.signature(b.fn).parameters.values())
            extra = [p.name for p in params[len(b.inputs) :]]
            assert set(extra) <= set(b.kwargs), (task.task_type, b.name, extra)
            assert set(b.kwargs) <= set(extra), (task.task_type, b.name)


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        TASKS["classification"] = None  # type: ignore[index]

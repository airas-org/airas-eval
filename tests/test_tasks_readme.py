"""The per-area task READMEs are generated; a stale one is a failing test."""

from airas_eval.tasks import AREAS
from airas_eval.tasks.readme import readme_path, render_area


def test_task_readmes_are_up_to_date():
    for area, tasks in AREAS.items():
        path = readme_path(area)
        assert path.exists(), f"missing {path}; run python -m airas_eval.tasks.readme"
        assert path.read_text() == render_area(area, tasks), (
            f"{path} is stale; run python -m airas_eval.tasks.readme"
        )


def test_every_registered_task_is_listed_in_exactly_one_area():
    from airas_eval.tasks import TASKS

    listed = [t.task_type for tasks in AREAS.values() for t in tasks]
    assert sorted(listed) == sorted(TASKS)
    assert len(listed) == len(set(listed))

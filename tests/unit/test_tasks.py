"""Tests for the bounded task DAG: size bound, uniqueness, acyclicity."""

import pytest
from pydantic import ValidationError

from personalos.domain.tasks import MAX_TASKS_PER_DAG, TaskDAG, TaskNode


def test_valid_linear_dag():
    dag = TaskDAG(
        goal="job_search",
        tasks=(
            TaskNode(id="prepare", name="Prepare"),
            TaskNode(id="search", name="Search", depends_on=("prepare",)),
        ),
    )
    assert dag.topological_order() == ["prepare", "search"]


def test_empty_dag_is_rejected():
    with pytest.raises(ValidationError):
        TaskDAG(goal="empty", tasks=())


def test_dag_over_the_bound_is_rejected():
    tasks = tuple(TaskNode(id=f"t{i}", name=f"Task {i}") for i in range(MAX_TASKS_PER_DAG + 1))
    with pytest.raises(ValidationError):
        TaskDAG(goal="too big", tasks=tasks)


def test_duplicate_task_ids_are_rejected():
    with pytest.raises(ValidationError):
        TaskDAG(
            goal="dup",
            tasks=(TaskNode(id="a", name="A"), TaskNode(id="a", name="A again")),
        )


def test_dependency_on_unknown_task_is_rejected():
    with pytest.raises(ValidationError):
        TaskDAG(
            goal="dangling",
            tasks=(TaskNode(id="a", name="A", depends_on=("missing",)),),
        )


def test_cycle_is_rejected():
    with pytest.raises(ValidationError):
        TaskDAG(
            goal="cycle",
            tasks=(
                TaskNode(id="a", name="A", depends_on=("b",)),
                TaskNode(id="b", name="B", depends_on=("a",)),
            ),
        )


def test_self_dependency_is_a_cycle():
    with pytest.raises(ValidationError):
        TaskDAG(goal="self", tasks=(TaskNode(id="a", name="A", depends_on=("a",)),))

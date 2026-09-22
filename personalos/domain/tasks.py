"""Bounded task DAG for graph-level planning.

A `TaskDAG` is a finite, acyclic plan: a fixed set of named steps with
explicit dependencies. It exists so a graph's planning step (e.g. the
Supervisor's `plan_work`) commits to a bounded set of work up front instead
of an open-ended agent loop that could keep proposing steps indefinitely.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed

#: Hard cap on the number of tasks in a single DAG. Planning that needs more
#: steps than this belongs in a longer-running, checkpointed workflow, not a
#: single bounded plan.
MAX_TASKS_PER_DAG = 12


class InvalidTaskDAG(ValidationFailed, ValueError):
    """A task DAG is empty, too large, cyclic, or references an unknown task.

    Subclasses both `ValidationFailed` (reports through the shared error
    taxonomy) and `ValueError`, matching `InvalidIdempotencyKey` and
    `UnsupportedRouteDomain` elsewhere in `personalos.domain`.
    """


class TaskNode(BaseModel):
    """One bounded unit of work in a plan."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    depends_on: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("id", "name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value


class TaskDAG(BaseModel):
    """A finite, acyclic plan: a goal and the bounded set of tasks to reach it."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: tuple[TaskNode, ...]

    @field_validator("tasks")
    @classmethod
    def _bounded_and_acyclic(cls, tasks: tuple[TaskNode, ...]) -> tuple[TaskNode, ...]:
        if not tasks:
            raise InvalidTaskDAG("a task DAG must contain at least one task")
        if len(tasks) > MAX_TASKS_PER_DAG:
            raise InvalidTaskDAG(
                f"task DAG has {len(tasks)} tasks, exceeding the bound of " f"{MAX_TASKS_PER_DAG}"
            )
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise InvalidTaskDAG("task ids must be unique")
        known_ids = set(ids)
        for task in tasks:
            unknown = set(task.depends_on) - known_ids
            if unknown:
                raise InvalidTaskDAG(
                    f"task '{task.id}' depends on unknown task(s): {sorted(unknown)}"
                )
        _topological_order(tasks)  # raises InvalidTaskDAG if a cycle exists
        return tasks

    def topological_order(self) -> list[str]:
        """Task ids ordered so every dependency precedes its dependents."""
        return _topological_order(self.tasks)


def _topological_order(tasks: tuple[TaskNode, ...]) -> list[str]:
    """Kahn's algorithm; raises InvalidTaskDAG if a dependency cycle remains."""
    remaining = {task.id: set(task.depends_on) for task in tasks}
    ordered: list[str] = []
    while remaining:
        ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
        if not ready:
            raise InvalidTaskDAG(f"task DAG has a dependency cycle among: {sorted(remaining)}")
        for task_id in ready:
            del remaining[task_id]
        for deps in remaining.values():
            deps.difference_update(ready)
        ordered.extend(ready)
    return ordered


__all__ = [
    "MAX_TASKS_PER_DAG",
    "InvalidTaskDAG",
    "TaskNode",
    "TaskDAG",
]

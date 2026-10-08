"""Reads a workflow's status out of the database, with no graph in hand.

`GET /v1/workflows/{id}` has to be answerable by any API process, including one
that booted after the workflow started and has never compiled the graph it runs
on. So everything here comes from rows: `workflows` / `workflow_runs` for which
threads exist and what the runner last said about them, `checkpoints` for
where each thread stopped, `checkpoint_writes` for the interrupts and errors
recorded against that stop, and the command queue for what has been handed to
the worker and not yet picked up.

Two facts are read out of LangGraph's checkpoint format rather than asked of a
compiled graph, and both are confined to this module:

- **Finished nodes** are the keys of `versions_seen`: a node appears there once
  its writes have been applied, in the order it first finished.
- **Next nodes** are the `branch:to:<node>` trigger channels the checkpoint
  still holds and the node has not yet consumed. That is the same test
  LangGraph's own scheduler applies, minus the graph topology it would use to
  explain *why* -- which a status read does not need.

An interrupt or an error is a write against the checkpoint the thread stopped
at, and its `task_path` names the node that raised it; that is how
`pending_approval` and `recoverable_failures` say *where*, not just *what*.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import CheckpointTuple
from sqlalchemy.orm import Session

from personalos.domain.workflow import (
    PendingInterrupt,
    RecoverableFailure,
    ThreadSnapshot,
    WorkflowSnapshot,
    thread_namespace,
)
from personalos.persistence.checkpointer import SqlAlchemyCheckpointSaver
from personalos.persistence.models import CheckpointWriteModel, WorkflowModel, WorkflowRunModel
from personalos.persistence.workflow_commands import WorkflowCommandQueue

logger = logging.getLogger(__name__)

#: Reserved write channels, as LangGraph spells them in `WRITES_IDX_MAP`.
_INTERRUPT_CHANNEL = "__interrupt__"
_ERROR_CHANNEL = "__error__"

#: Prefix of the per-node trigger channels a LangGraph checkpoint carries.
_TRIGGER_PREFIX = "branch:to:"

#: Prefix LangGraph puts on the path of a task pulled from a trigger channel,
#: e.g. `"~__pregel_pull, approval_checkpoint"`.
_PULL_TASK_PREFIX = "~__pregel_pull, "


class WorkflowStatusReader:
    """Builds a `WorkflowSnapshot` from stored rows, for any process to serve."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        checkpointer: SqlAlchemyCheckpointSaver,
        commands: WorkflowCommandQueue,
    ):
        """Initialize with the session factory, saver and queue the workflow lives in."""
        self.session_factory = session_factory
        self.checkpointer = checkpointer
        self.commands = commands

    def read(self, workflow_id: UUID) -> WorkflowSnapshot | None:
        """The workflow as the database has it, or `None` if it does not exist."""
        session = self.session_factory()
        try:
            workflow = session.query(WorkflowModel).filter(WorkflowModel.id == workflow_id).first()
            if workflow is None:
                return None
            runs = (
                session.query(WorkflowRunModel)
                .filter(WorkflowRunModel.workflow_id == workflow_id)
                .order_by(WorkflowRunModel.created_at.asc(), WorkflowRunModel.thread_id.asc())
                .all()
            )
            threads = tuple(self._thread(session, run) for run in runs)
            name, created_at = workflow.name, workflow.created_at
        finally:
            session.close()

        return WorkflowSnapshot(
            workflow_id=workflow_id,
            name=name,
            threads=threads,
            queued=tuple(self.commands.queued_for(workflow_id)),
            created_at=created_at,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _thread(self, session: Session, run: WorkflowRunModel) -> ThreadSnapshot:
        """Snapshot one thread from its run row and its newest checkpoint."""
        stored = self.checkpointer.get_tuple({"configurable": {"thread_id": run.thread_id}})
        base = {
            "thread_id": run.thread_id,
            "kind": thread_namespace(run.thread_id),
            "run_status": run.status,
            "actor_id": run.actor_id,
            "correlation_id": run.correlation_id,
            "updated_at": run.updated_at,
        }
        if stored is None:
            return ThreadSnapshot(**base)

        checkpoint = stored.checkpoint
        checkpoint_id = stored.config["configurable"]["checkpoint_id"]
        steps_by_task = self._steps_by_task(session, run.thread_id, checkpoint_id)
        return ThreadSnapshot(
            **base,
            checkpoint_id=checkpoint_id,
            completed_steps=_completed_steps(checkpoint.get("versions_seen") or {}),
            next_steps=_next_steps(checkpoint),
            interrupts=_interrupts(run.thread_id, stored, steps_by_task),
            failures=_failures(run.thread_id, stored, steps_by_task),
        )

    @staticmethod
    def _steps_by_task(session: Session, thread_id: str, checkpoint_id: str) -> dict[str, str]:
        """Map each task that wrote against a checkpoint to the node it ran.

        Read from the rows directly because `CheckpointTuple.pending_writes`
        does not carry the task path, and the task path is the only place the
        node name survives.
        """
        rows = (
            session.query(CheckpointWriteModel.task_id, CheckpointWriteModel.task_path)
            .filter(
                CheckpointWriteModel.thread_id == thread_id,
                CheckpointWriteModel.checkpoint_ns == "",
                CheckpointWriteModel.checkpoint_id == checkpoint_id,
            )
            .all()
        )
        steps: dict[str, str] = {}
        for task_id, task_path in rows:
            step = _step_from_task_path(task_path)
            if step:
                steps[task_id] = step
        return steps


def _step_from_task_path(task_path: str | None) -> str | None:
    """The node name in a pulled task's path, or `None` for any other shape."""
    if task_path and task_path.startswith(_PULL_TASK_PREFIX):
        return task_path[len(_PULL_TASK_PREFIX) :].strip() or None
    return None


def _is_internal(name: str) -> bool:
    return name.startswith("__")


def _completed_steps(versions_seen: Mapping[str, Any]) -> tuple[str, ...]:
    """Nodes whose writes have been applied, in first-finished order."""
    return tuple(name for name in versions_seen if not _is_internal(name))


def _next_steps(checkpoint: Mapping[str, Any]) -> tuple[str, ...]:
    """Nodes with a trigger the checkpoint holds and the node has not consumed."""
    values = checkpoint.get("channel_values") or {}
    versions = checkpoint.get("channel_versions") or {}
    seen = checkpoint.get("versions_seen") or {}
    pending: list[str] = []
    for channel in values:
        if not channel.startswith(_TRIGGER_PREFIX):
            continue
        node = channel[len(_TRIGGER_PREFIX) :]
        if _is_internal(node):
            continue
        current = versions.get(channel)
        consumed = (seen.get(node) or {}).get(channel)
        try:
            unconsumed = consumed is None or (current is not None and current > consumed)
        except TypeError:  # pragma: no cover - mixed version types; trust the presence
            unconsumed = True
        if unconsumed:
            pending.append(node)
    return tuple(pending)


def _interrupts(
    thread_id: str, stored: CheckpointTuple, steps_by_task: Mapping[str, str]
) -> tuple[PendingInterrupt, ...]:
    """The interrupts recorded against a thread's newest checkpoint."""
    found: list[PendingInterrupt] = []
    for task_id, channel, value in stored.pending_writes or ():
        if channel != _INTERRUPT_CHANNEL:
            continue
        for interrupt in value if isinstance(value, list | tuple) else (value,):
            found.append(
                PendingInterrupt(
                    thread_id=thread_id,
                    step=steps_by_task.get(task_id),
                    interrupt_id=getattr(interrupt, "id", None),
                    value=getattr(interrupt, "value", interrupt),
                )
            )
    return tuple(found)


def _failures(
    thread_id: str, stored: CheckpointTuple, steps_by_task: Mapping[str, str]
) -> tuple[RecoverableFailure, ...]:
    """Node errors recorded against a thread's newest checkpoint.

    Recoverable by construction: the checkpoint they were recorded against is
    intact, and a resume re-runs exactly the node that raised. The error text
    is what LangGraph stored -- already passed through the saver's redactor.
    """
    return tuple(
        RecoverableFailure(thread_id=thread_id, step=steps_by_task.get(task_id), message=str(value))
        for task_id, channel, value in stored.pending_writes or ()
        if channel == _ERROR_CHANNEL
    )


__all__ = ["WorkflowStatusReader"]

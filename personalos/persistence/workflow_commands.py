"""The queue the API hands workflow runs to, instead of running them itself.

A run can take minutes (provider searches, model calls) and can park for days
at an approval, so a request handler that ran it inline would either block for
that long or hold the run in a process that might not be there to finish it.
The API therefore writes a `WorkflowCommand` here and returns; a worker claims
it and runs it under the workflow's lease.

Commands are rows in `outbox_events` -- the same transactional outbox the rest
of the system already relays through -- typed `workflow.command`. Two
properties come from that table and are relied on:

- **A claim is exclusive.** `OutboxEventRepository.claim_next` is `SKIP LOCKED`
  on Postgres and a guarded `UPDATE` elsewhere, so two workers never run the
  same command.
- **`dedupe_key` is unique.** Each command's key starts with its workflow id
  (`workflow:<id>:...`), which is how `queued_for` finds a workflow's
  unfinished commands without a payload query, and a resume's key names the
  checkpoint it answers, which is how a second answer to the same interrupt is
  refused at the database rather than queued behind the first.
"""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.errors import Conflict
from personalos.domain.models import OutboxEventStatus
from personalos.domain.workflow import WorkflowCommand
from personalos.persistence.models import OutboxEventModel
from personalos.persistence.repositories import OutboxEventRepository

#: `outbox_events.type` for every command this queue carries.
WORKFLOW_COMMAND_EVENT = "workflow.command"

#: Outbox statuses a command counts as still queued in: not yet claimed, or
#: claimed by a worker that has not finished dispatching it.
_UNFINISHED = (OutboxEventStatus.PENDING.value, OutboxEventStatus.IN_PROGRESS.value)


class DuplicateWorkflowCommand(Conflict):
    """The same command -- typically the same resume -- is already queued."""

    default_message = "an equivalent command for this workflow is already queued"


@dataclass(frozen=True)
class ClaimedWorkflowCommand:
    """A command a worker now owns, with the row it must settle."""

    outbox_id: UUID
    command: WorkflowCommand


def _key_prefix(workflow_id: UUID) -> str:
    return f"workflow:{workflow_id}:"


class WorkflowCommandQueue:
    """Enqueues, lists and claims `WorkflowCommand`s in `outbox_events`."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    def enqueue(self, command: WorkflowCommand, *, dedupe: str | None = None) -> WorkflowCommand:
        """Queue a command for the worker.

        `dedupe` is what makes two commands "the same" -- for a resume, the
        checkpoint it answers. Left out, the command id is used, and every
        command is distinct. Raises `DuplicateWorkflowCommand` when an
        equivalent one was queued before.
        """
        key = (
            f"{_key_prefix(command.workflow_id)}{command.kind.value}:{dedupe or command.command_id}"
        )
        session = self.session_factory()
        try:
            try:
                OutboxEventRepository(session).create(
                    type=WORKFLOW_COMMAND_EVENT,
                    payload=command.model_dump(mode="json"),
                    dedupe_key=key,
                )
            except IntegrityError as exc:
                session.rollback()
                raise DuplicateWorkflowCommand(
                    f"a {command.kind.value} command for workflow {command.workflow_id} "
                    f"answering the same state is already queued",
                    details={"workflow_id": str(command.workflow_id)},
                ) from exc
            return command
        finally:
            session.close()

    def queued_for(self, workflow_id: UUID) -> list[WorkflowCommand]:
        """A workflow's commands no worker has finished dispatching, oldest first."""
        session = self.session_factory()
        try:
            rows = (
                session.query(OutboxEventModel)
                .filter(
                    OutboxEventModel.type == WORKFLOW_COMMAND_EVENT,
                    OutboxEventModel.dedupe_key.like(f"{_key_prefix(workflow_id)}%"),
                    OutboxEventModel.status.in_(_UNFINISHED),
                )
                .order_by(OutboxEventModel.created_at.asc(), OutboxEventModel.id.asc())
                .all()
            )
            return [self._command(row) for row in rows]
        finally:
            session.close()

    def claim_next(self) -> ClaimedWorkflowCommand | None:
        """Claim the oldest queued command for this worker, or `None` if there is none."""
        session = self.session_factory()
        try:
            row = OutboxEventRepository(session).claim_next(type=WORKFLOW_COMMAND_EVENT)
            if row is None:
                return None
            return ClaimedWorkflowCommand(outbox_id=row.id, command=self._command(row))
        finally:
            session.close()

    def mark_done(self, outbox_id: UUID) -> None:
        """Record that a claimed command was run (to completion or to a pause)."""
        session = self.session_factory()
        try:
            OutboxEventRepository(session).mark_dispatched(outbox_id)
        finally:
            session.close()

    def release(self, outbox_id: UUID) -> None:
        """Put a claimed command back in line, for a worker that could not run it yet."""
        session = self.session_factory()
        try:
            session.query(OutboxEventModel).filter(
                OutboxEventModel.id == outbox_id,
                OutboxEventModel.status == OutboxEventStatus.IN_PROGRESS.value,
            ).update(
                {OutboxEventModel.status: OutboxEventStatus.PENDING.value},
                synchronize_session=False,
            )
            session.commit()
        finally:
            session.close()

    def mark_failed(self, outbox_id: UUID) -> None:
        """Record that running a claimed command raised."""
        session = self.session_factory()
        try:
            OutboxEventRepository(session).mark_failed(outbox_id)
        finally:
            session.close()

    @staticmethod
    def _command(row: OutboxEventModel) -> WorkflowCommand:
        return WorkflowCommand.model_validate({**row.payload_json, "created_at": row.created_at})


__all__ = [
    "WORKFLOW_COMMAND_EVENT",
    "DuplicateWorkflowCommand",
    "ClaimedWorkflowCommand",
    "WorkflowCommandQueue",
]

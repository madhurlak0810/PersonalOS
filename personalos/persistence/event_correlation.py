"""Which waiting workflow, if any, an incoming event belongs to.

An event resumes a workflow rather than starting a new run when that workflow
is parked on an interrupt *and* shares the event's `correlation_id`. The
correlation is stamped on each `workflow_runs` row when its thread is
registered, so finding the candidates is one indexed-by-value query; whether
one of them is actually waiting is the same question `GET /v1/workflows/{id}`
answers, and is answered by the same reader, so the two can never disagree
about what "waiting" means -- including that an interrupt whose answer is
already queued is not waiting any more.
"""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.persistence.models import WorkflowRunModel
from personalos.persistence.workflow_status import WorkflowStatusReader


@dataclass(frozen=True)
class WaitingThread:
    """A thread parked on an interrupt that an event can answer."""

    workflow_id: UUID
    thread_id: str
    #: The interrupts on that thread a resume would answer.
    interrupt_ids: tuple[str, ...]
    actor_id: str


class CorrelatedWaitLookup:
    """Finds the waiting thread correlated with an event."""

    def __init__(self, session_factory: Callable[[], Session], status_reader: WorkflowStatusReader):
        """Initialize with the session factory and the reader that defines "waiting"."""
        self.session_factory = session_factory
        self.status_reader = status_reader

    def waiting_for(self, correlation_id: UUID) -> WaitingThread | None:
        """The oldest workflow under `correlation_id` with an unanswered interrupt."""
        session = self.session_factory()
        try:
            rows = (
                session.query(WorkflowRunModel.workflow_id, WorkflowRunModel.created_at)
                .filter(WorkflowRunModel.correlation_id == correlation_id)
                .order_by(WorkflowRunModel.created_at.asc())
                .all()
            )
        finally:
            session.close()

        seen: set[UUID] = set()
        for workflow_id, _created_at in rows:
            if workflow_id in seen:
                continue
            seen.add(workflow_id)
            snapshot = self.status_reader.read(workflow_id)
            if snapshot is None or not snapshot.pending_interrupts:
                continue
            thread_id = snapshot.pending_interrupts[0].thread_id
            thread = next(t for t in snapshot.threads if t.thread_id == thread_id)
            return WaitingThread(
                workflow_id=workflow_id,
                thread_id=thread_id,
                interrupt_ids=tuple(
                    interrupt.interrupt_id
                    for interrupt in snapshot.pending_interrupts
                    if interrupt.thread_id == thread_id and interrupt.interrupt_id
                ),
                actor_id=thread.actor_id or "system",
            )
        return None


__all__ = ["CorrelatedWaitLookup", "WaitingThread"]

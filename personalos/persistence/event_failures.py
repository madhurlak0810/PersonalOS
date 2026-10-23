"""Outbox events the worker gave up on, as failures an operator can see and retry.

A dead-lettered or quarantined `outbox_events` row keeps its last error and,
when it could be told, the workflow it belongs to. This module reads those
rows as `EventFailure`s -- all of them, or one workflow's for
`GET /v1/workflows/{id}/failures` -- and puts one back in line on request,
with an `audit_events` row naming who did it.
"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.events import ENVELOPE_KEY
from personalos.domain.failures import EventFailure, assess_event_failure
from personalos.domain.models import AuditEventResult, OutboxEventStatus
from personalos.persistence.event_outbox import EventOutbox
from personalos.persistence.models import OutboxEventModel
from personalos.persistence.repositories import AuditEventRepository

#: `audit_events.action` for an operator's retry of a parked event.
EVENT_RETRY_REQUESTED = "outbox.event.retry_requested"

#: The statuses listed: the two the dispatcher parks rows in. `failed` rows
#: predate them and are left out of the view, though `retry` accepts them.
LISTED_STATUSES = (OutboxEventStatus.DEAD_LETTER.value, OutboxEventStatus.QUARANTINED.value)

#: Most rows one listing returns, newest failure first.
DEFAULT_LIMIT = 100


def event_target(outbox_id: UUID) -> str:
    """`audit_events.target_ref` for an event about one outbox row."""
    return f"outbox_event:{outbox_id}"


class EventFailureStore:
    """Lists parked outbox events and retries them."""

    def __init__(self, session_factory: Callable[[], Session], outbox: EventOutbox):
        """Initialize over the database and the outbox that requeues."""
        self.session_factory = session_factory
        self.outbox = outbox

    def failures(
        self, *, workflow_id: UUID | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[EventFailure]:
        """Parked events, newest failure first; only `workflow_id`'s when given."""
        session = self.session_factory()
        try:
            query = session.query(OutboxEventModel).filter(
                OutboxEventModel.status.in_(LISTED_STATUSES)
            )
            if workflow_id is not None:
                query = query.filter(OutboxEventModel.workflow_id == workflow_id)
            rows = (
                query.order_by(OutboxEventModel.failed_at.desc(), OutboxEventModel.id)
                .limit(limit)
                .all()
            )
            return [_assess(row) for row in rows]
        finally:
            session.close()

    def exists(self, outbox_id: UUID) -> bool:
        """Whether the outbox has a row with this id, in any status."""
        session = self.session_factory()
        try:
            return session.get(OutboxEventModel, outbox_id) is not None
        finally:
            session.close()

    def retry(self, outbox_id: UUID, *, actor: str) -> OutboxEventModel | None:
        """Put a parked event back in line, auditing who did. `None` if it is not parked."""
        row = self.outbox.requeue(outbox_id)
        if row is None:
            return None
        session = self.session_factory()
        try:
            AuditEventRepository(session).create(
                actor=actor,
                action=EVENT_RETRY_REQUESTED,
                target_ref=event_target(outbox_id),
                result=AuditEventResult.SUCCESS.value,
                workflow_id=row.workflow_id,
            )
        finally:
            session.close()
        return row


def _assess(row: OutboxEventModel) -> EventFailure:
    stored = (row.payload_json or {}).get(ENVELOPE_KEY)
    event_id = None
    if isinstance(stored, dict):
        try:
            event_id = UUID(str(stored.get("event_id")))
        except (TypeError, ValueError):
            event_id = None
    return assess_event_failure(
        outbox_id=row.id,
        event_id=event_id,
        event_type=row.type,
        status=row.status,
        last_error=row.last_error,
        attempts=row.attempts,
        dedupe_key=row.dedupe_key,
        workflow_id=row.workflow_id,
        created_at=row.created_at,
        failed_at=row.failed_at,
    )


__all__ = [
    "DEFAULT_LIMIT",
    "EVENT_RETRY_REQUESTED",
    "LISTED_STATUSES",
    "EventFailureStore",
    "event_target",
]

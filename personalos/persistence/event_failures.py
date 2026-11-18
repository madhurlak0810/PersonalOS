"""Outbox events the worker gave up on, as failures an operator can see and retry.

A dead-lettered or quarantined `outbox_events` row keeps its last error and,
when it could be told, the workflow it belongs to. This module reads those
rows as `EventFailure`s -- all of them, or one workflow's for
`GET /v1/workflows/{id}/failures` -- and puts one back in line on request,
with an `audit_events` row naming who did it.

Every read and the retry are scoped: a scoped caller sees the parked events of
its own workflows, and -- only with `include_unowned`, which the API passes for
an `admin` token -- the install-wide ones that belong to no workflow.
"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.events import ENVELOPE_KEY
from personalos.domain.failures import EventFailure, assess_event_failure
from personalos.domain.models import AuditEventResult, OutboxEventStatus
from personalos.domain.ownership import Access, owner_of
from personalos.persistence.event_outbox import EventOutbox
from personalos.persistence.models import OutboxEventModel
from personalos.persistence.ownership import (
    ObjectType,
    deny_if_not_visible,
    owned_workflow_ids,
    workflow_visible,
)
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
        self,
        *,
        access: Access,
        include_unowned: bool = False,
        workflow_id: UUID | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[EventFailure]:
        """Parked events `access` may see, newest failure first; only `workflow_id`'s when given."""
        session = self.session_factory()
        try:
            query = _visible(
                session.query(OutboxEventModel).filter(
                    OutboxEventModel.status.in_(LISTED_STATUSES)
                ),
                access,
                include_unowned,
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

    def exists(self, outbox_id: UUID, *, access: Access, include_unowned: bool = False) -> bool:
        """Whether the outbox has a row with this id `access` may see, in any status."""
        session = self.session_factory()
        try:
            row = session.get(OutboxEventModel, outbox_id)
            return row is not None and _can_see(session, row, access, include_unowned)
        finally:
            session.close()

    def retry(
        self,
        outbox_id: UUID,
        *,
        actor: str,
        access: Access,
        include_unowned: bool = False,
    ) -> OutboxEventModel | None:
        """Put a parked event back in line, auditing who did.

        `None` if it is not parked, or `access` cannot see it (audited).
        """
        if not deny_if_not_visible(
            self.session_factory,
            access,
            visible=self.exists(outbox_id, access=access, include_unowned=include_unowned),
            object_type=ObjectType.EVENT_FAILURE,
            object_id=outbox_id,
        ):
            return None
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


def _visible(query, access: Access, include_unowned: bool):
    owned = owned_workflow_ids(access)
    if owned is None:
        return query
    mine = OutboxEventModel.workflow_id.in_(owned)
    if include_unowned:
        return query.filter(mine | OutboxEventModel.workflow_id.is_(None))
    return query.filter(mine)


def _can_see(
    session: Session, row: OutboxEventModel, access: Access, include_unowned: bool
) -> bool:
    if owner_of(access) is None:
        return True
    if row.workflow_id is None:
        return include_unowned
    return workflow_visible(session, row.workflow_id, access)


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

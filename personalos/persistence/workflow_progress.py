"""A workflow's progress events, kept in `event_log` and read back in order.

The write side is `record`, called by the worker running the workflow. The
read side is `events_after`, called by `GET /v1/workflows/{id}/events` in any
API process. The two share nothing but rows, so a client gets the same events
whether or not it was connected while they happened.

**Order is the log's, not the clock's.** A stream resumes from `Last-Event-ID`
by asking for what comes after that row in `(occurred_at, id)` order. That is
only exact if no row is ever written *behind* one a client has already seen,
so `record` never stamps an event earlier than the newest one already on the
workflow: it takes `max(now, newest + 1µs)`. A worker whose clock runs behind
the previous lease holder's still appends after it. Every recorder of a
workflow's events runs under that workflow's lease, so there is one writer at
a time and the read-then-append cannot race.

**Recording never fails a run.** A progress event is an observation of a
step, not part of it: if the log cannot be written, the failure is logged and
the step carries on. Nothing waits on a reader either -- an event is one
committed insert, there is no subscriber to deliver it to.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from personalos.domain.progress import (
    DEFAULT_PROGRESS_BATCH,
    STORED_PROGRESS_TYPES,
    WORKFLOW_AGGREGATE,
    InvalidEventCursor,
    ProgressEventType,
    WorkflowProgressEvent,
)
from personalos.domain.redaction import redact
from personalos.persistence.models import EventLogModel, WorkflowModel
from personalos.persistence.repositories import EventLogRepository

logger = logging.getLogger(__name__)

#: The smallest step `record` moves a timestamp forward by to keep it after
#: the newest event; the resolution both SQLite and Postgres keep.
_TICK = timedelta(microseconds=1)


class WorkflowProgressLog:
    """Appends and reads a workflow's progress events in `event_log`."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def record(
        self,
        workflow_id: UUID,
        kind: ProgressEventType,
        *,
        thread_id: str | None = None,
        correlation_id: UUID | str | None = None,
        data: dict[str, Any] | None = None,
    ) -> UUID | None:
        """Append one event; its id, or `None` if it could not be written.

        Never raises: see the module docstring.
        """
        payload = {
            "thread_id": thread_id,
            "correlation_id": str(correlation_id) if correlation_id else None,
            "data": data or {},
        }
        session = self.session_factory()
        try:
            row = EventLogRepository(session).append(
                aggregate_type=WORKFLOW_AGGREGATE,
                aggregate_id=workflow_id,
                event_type=kind.stored,
                payload=payload,
                occurred_at=self._next_timestamp(session, workflow_id),
            )
            return row.id
        except Exception:
            session.rollback()
            logger.exception(
                "could not record %s for workflow %s; the run carries on without it",
                kind.value,
                workflow_id,
            )
            return None
        finally:
            session.close()

    @staticmethod
    def _next_timestamp(session: Session, workflow_id: UUID) -> datetime:
        now = datetime.utcnow()
        newest = (
            session.query(EventLogModel.occurred_at)
            .filter(
                EventLogModel.aggregate_type == WORKFLOW_AGGREGATE,
                EventLogModel.aggregate_id == workflow_id,
            )
            .order_by(EventLogModel.occurred_at.desc())
            .limit(1)
            .scalar()
        )
        return now if newest is None or now > newest else newest + _TICK

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def workflow_exists(self, workflow_id: UUID) -> bool:
        """Whether the workflow is registered at all."""
        session = self.session_factory()
        try:
            return (
                session.query(WorkflowModel.id).filter(WorkflowModel.id == workflow_id).first()
                is not None
            )
        finally:
            session.close()

    def events_after(
        self,
        workflow_id: UUID,
        after: UUID | None = None,
        *,
        limit: int = DEFAULT_PROGRESS_BATCH,
    ) -> list[WorkflowProgressEvent]:
        """The workflow's events after `after` (from the first when `None`), oldest first.

        Raises `InvalidEventCursor` when `after` is not one of this workflow's
        events. At most `limit`; call again from the last one for the rest.
        """
        session = self.session_factory()
        try:
            query = session.query(EventLogModel).filter(
                EventLogModel.aggregate_type == WORKFLOW_AGGREGATE,
                EventLogModel.aggregate_id == workflow_id,
                EventLogModel.event_type.in_(STORED_PROGRESS_TYPES),
            )
            if after is not None:
                cursor = (
                    session.query(EventLogModel.occurred_at, EventLogModel.id)
                    .filter(
                        EventLogModel.id == after,
                        EventLogModel.aggregate_type == WORKFLOW_AGGREGATE,
                        EventLogModel.aggregate_id == workflow_id,
                    )
                    .first()
                )
                if cursor is None:
                    raise InvalidEventCursor(
                        f"'{after}' is not an event of workflow {workflow_id}; "
                        f"reconnect without Last-Event-ID to read from the start",
                        details={"workflow_id": str(workflow_id), "last_event_id": str(after)},
                    )
                at, row_id = cursor
                query = query.filter(
                    or_(
                        EventLogModel.occurred_at > at,
                        and_(EventLogModel.occurred_at == at, EventLogModel.id > row_id),
                    )
                )
            rows = (
                query.order_by(EventLogModel.occurred_at.asc(), EventLogModel.id.asc())
                .limit(limit)
                .all()
            )
            return [event for event in map(_to_event, rows) if event is not None]
        finally:
            session.close()


def _to_event(row: EventLogModel) -> WorkflowProgressEvent | None:
    kind = ProgressEventType.from_stored(row.event_type)
    if kind is None:  # pragma: no cover - filtered out by the query
        return None
    # Redacted again on the way out, as the audit trail is: a secret the
    # redactor learned of after the row was written is still kept back.
    payload = redact(row.payload_json or {})
    correlation_id = payload.get("correlation_id")
    try:
        correlation = UUID(str(correlation_id)) if correlation_id else None
    except ValueError:
        correlation = None
    return WorkflowProgressEvent(
        event_id=row.id,
        type=kind,
        workflow_id=row.aggregate_id,
        correlation_id=correlation,
        thread_id=payload.get("thread_id"),
        at=row.occurred_at,
        data=payload.get("data") or {},
    )


__all__ = ["WorkflowProgressLog"]

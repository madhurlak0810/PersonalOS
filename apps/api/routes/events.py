"""A workflow's progress, streamed. Version 1.

- `GET /v1/workflows/{id}/events` is a Server-Sent Events stream of the
  workflow's progress events: `step_started`, `step_completed`,
  `approval_requested`, `approval_resolved`, `failure`, `workflow_completed`.
  Each frame's `id:` is the event's id, its `event:` the kind, and its `data:`
  the event as JSON, `workflow_id` and `correlation_id` included.

**The stream is a read of the log, not a subscription.** Events are rows of
`event_log` that the worker appended as the workflow ran
(`personalos.persistence.workflow_progress`). This endpoint reads them in
order and then polls for more, so it works from any API process, and a
workflow that ran while nobody watched has lost nothing.

**Reconnecting is exact.** A client that sends `Last-Event-ID` (as a browser
`EventSource` does on its own when a connection drops) gets every event after
that one and nothing before it: no gap, no repeat. `?last_event_id=` does the
same for a client that cannot set headers. An id that is not one of this
workflow's events is a 422 rather than a replay from the start.

**Nothing here touches the run.** The stream only reads; the worker only
appends. A slow or vanished client holds up nothing but its own connection.

`?follow=false` sends what has happened so far and closes, instead of
waiting for more: one read of the backlog, for a client that would rather
poll.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from personalos.bootstrap import build_workflow_progress_log
from personalos.domain.errors import NotFound, ValidationFailed
from personalos.domain.progress import DEFAULT_PROGRESS_BATCH, WorkflowProgressEvent
from personalos.persistence.database import SessionLocal
from personalos.persistence.workflow_progress import WorkflowProgressLog

router = APIRouter()

#: The SSE media type, and the headers that stop a proxy buffering the stream.
EVENT_STREAM = "text/event-stream"
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# --- Dependencies ---------------------------------------------------------------


@dataclass(frozen=True)
class EventStreamSettings:
    """How often a following stream looks for new events, and keeps itself alive."""

    poll_seconds: float = 1.0
    #: A comment frame after this long with nothing to send, so an idle
    #: connection is not cut by a proxy that thinks it is dead.
    heartbeat_seconds: float = 15.0
    #: The reconnect delay suggested to the client (the SSE `retry:` field).
    retry_ms: int = 3000
    batch_size: int = DEFAULT_PROGRESS_BATCH


def get_event_stream_settings() -> EventStreamSettings:
    """FastAPI dependency: the default stream settings."""
    return EventStreamSettings()


def build_progress_log(session_factory: Callable[[], Any]) -> WorkflowProgressLog:
    """The progress log over one database."""
    return build_workflow_progress_log(session_factory)


def get_progress_log() -> WorkflowProgressLog:
    """FastAPI dependency: the progress log over the configured database."""
    return build_progress_log(SessionLocal)


# --- Framing --------------------------------------------------------------------


def sse_frame(event: WorkflowProgressEvent) -> str:
    """One event as an SSE frame."""
    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.type.value}\ndata: {data}\n\n"


async def progress_stream(
    log: WorkflowProgressLog,
    workflow_id: UUID,
    *,
    after: UUID | None,
    follow: bool,
    settings: EventStreamSettings,
    is_disconnected: Callable[[], Awaitable[bool]],
    first: list[WorkflowProgressEvent] | None = None,
) -> AsyncIterator[str]:
    """The frames of a workflow's events after `after`, then (if `follow`) of new ones.

    `first` is a batch already read from `after`, so the route can check the
    cursor before the response starts.
    """
    yield f"retry: {settings.retry_ms}\n\n"
    cursor = after
    idle = 0.0
    batch = first
    while True:
        if batch is None:
            batch = await run_in_threadpool(
                log.events_after, workflow_id, cursor, limit=settings.batch_size
            )
        for event in batch:
            yield sse_frame(event)
            cursor = event.event_id
        full = len(batch) >= settings.batch_size
        if batch:
            idle = 0.0
        batch = None
        if full:
            # More may be waiting right behind this batch; read on without pausing.
            continue
        if not follow or await is_disconnected():
            return
        await asyncio.sleep(settings.poll_seconds)
        idle += settings.poll_seconds
        if idle >= settings.heartbeat_seconds:
            yield ": keepalive\n\n"
            idle = 0.0


def _cursor(value: str | None) -> UUID | None:
    if value is None or not value.strip():
        return None
    try:
        return UUID(value.strip())
    except ValueError:
        raise ValidationFailed(
            f"Last-Event-ID '{value}' is not an event id",
            details={"last_event_id": value},
        ) from None


# --- Routes ---------------------------------------------------------------------


@router.get(
    "/workflows/{workflow_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {EVENT_STREAM: {}}}},
)
async def stream_workflow_events(
    workflow_id: UUID,
    request: Request,
    follow: bool = Query(default=True),
    last_event_id_query: str | None = Query(default=None, alias="last_event_id"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    log: WorkflowProgressLog = Depends(get_progress_log),
    settings: EventStreamSettings = Depends(get_event_stream_settings),
):
    """Stream the workflow's progress events, resuming after `Last-Event-ID` if given.

    The header wins over `?last_event_id=` when both are sent: it is what a
    reconnecting `EventSource` sets, and it is the newer of the two.
    """
    if not await run_in_threadpool(log.workflow_exists, workflow_id):
        raise NotFound(
            f"workflow '{workflow_id}' not found", details={"workflow_id": str(workflow_id)}
        )
    after = _cursor(last_event_id if last_event_id is not None else last_event_id_query)
    # Read the first batch here, before the response starts, so a cursor that
    # names no event of this workflow is an error response, not a broken stream.
    first = await run_in_threadpool(log.events_after, workflow_id, after, limit=settings.batch_size)
    return StreamingResponse(
        progress_stream(
            log,
            workflow_id,
            after=after,
            follow=follow,
            settings=settings,
            is_disconnected=request.is_disconnected,
            first=first,
        ),
        media_type=EVENT_STREAM,
        headers=_STREAM_HEADERS,
    )


__all__ = [
    "router",
    "EVENT_STREAM",
    "EventStreamSettings",
    "build_progress_log",
    "get_event_stream_settings",
    "get_progress_log",
    "progress_stream",
    "sse_frame",
]

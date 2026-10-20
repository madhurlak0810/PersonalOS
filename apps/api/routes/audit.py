"""The audit trail over HTTP: what happened, and why. Version 1.

Two endpoints, both read-only:

- `GET /v1/audit` searches the trail: filter by workflow, application, actor,
  action and time range, newest first, a page at a time.
- `GET /v1/workflows/{id}/audit` is one workflow's whole trail, oldest first:
  each action as intent -> policy decision -> approval -> execution, with the
  provider's receipt on the execution.

**Nothing here writes.** There is no POST, PUT, PATCH or DELETE on either
path, and the store behind them (`AuditTrailStore`) has no method that could
change an `audit_events` row. The trail is append-only on the write side; this
is the read side keeping it that way.

**Every entry carries its chain.** The `policy_decisions`, `tool_executions`
and `approvals` rows of the action an entry belongs to are attached under
`links`, so any one entry is enough to trace the rest.

**Payloads are redacted.** Every value has been through the redactor on its way
out, so a credential that reached a row -- a receipt that echoed a token, an
error carrying a header -- comes back as `[REDACTED]`.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from personalos.bootstrap import build_audit_trail_store
from personalos.domain.audit import (
    DEFAULT_AUDIT_PAGE_SIZE,
    MAX_AUDIT_PAGE_SIZE,
    AuditQuery,
    AuditTrailEntry,
)
from personalos.domain.errors import NotFound
from personalos.persistence.audit_trail import AuditTrailStore
from personalos.persistence.database import SessionLocal

router = APIRouter()


# --- Dependencies ---------------------------------------------------------------


def build_audit_store(session_factory: Callable[[], Any]) -> AuditTrailStore:
    """The audit store over one database."""
    return build_audit_trail_store(session_factory)


def get_audit_store() -> AuditTrailStore:
    """FastAPI dependency: a fresh store over the configured database."""
    return build_audit_store(SessionLocal)


# --- Response models ------------------------------------------------------------


class AuditSearchResponse(BaseModel):
    """One page of the audit trail, newest first."""

    entries: list[AuditTrailEntry] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    #: Pass as `offset` for the next page; absent on the last one.
    next_offset: int | None = None


class WorkflowAuditResponse(BaseModel):
    """One workflow's trail, oldest first."""

    workflow_id: UUID
    entries: list[AuditTrailEntry] = Field(default_factory=list)


# --- Routes ---------------------------------------------------------------------


@router.get("/audit", response_model=AuditSearchResponse)
async def search_audit(
    workflow_id: UUID | None = Query(default=None),
    application_id: UUID | None = Query(default=None),
    actor: str | None = Query(default=None, max_length=255),
    action: str | None = Query(default=None, max_length=255),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=DEFAULT_AUDIT_PAGE_SIZE, ge=1, le=MAX_AUDIT_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    store: AuditTrailStore = Depends(get_audit_store),
):
    """Audit events matching every filter given, newest first.

    `application_id` matches the workflows run on that application's own
    thread -- the board moves made on it. `since` and `until` are inclusive.
    """
    page = store.search(
        AuditQuery(
            workflow_id=workflow_id,
            application_id=application_id,
            actor=actor or None,
            action=action or None,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    )
    return AuditSearchResponse(
        entries=list(page.entries),
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        next_offset=page.next_offset,
    )


@router.get("/workflows/{workflow_id}/audit", response_model=WorkflowAuditResponse)
async def get_workflow_audit(
    workflow_id: UUID,
    store: AuditTrailStore = Depends(get_audit_store),
):
    """Everything recorded for one workflow, in the order it happened."""
    trail = store.workflow_trail(workflow_id)
    if trail is None:
        raise NotFound(
            f"workflow '{workflow_id}' not found",
            details={"workflow_id": str(workflow_id)},
        )
    return WorkflowAuditResponse(workflow_id=trail.workflow_id, entries=list(trail.entries))


__all__ = [
    "router",
    "AuditSearchResponse",
    "WorkflowAuditResponse",
    "build_audit_store",
    "get_audit_store",
]

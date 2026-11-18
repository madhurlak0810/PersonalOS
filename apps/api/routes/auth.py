"""Who the caller is, and tickets for the event stream. Version 1.

- `GET /v1/auth/whoami` is the verified principal: the actor every route
  records for this caller, and how it was established. A UI calls it to check
  its token before showing anything.
- `POST /v1/auth/stream-tickets` exchanges the bearer token for a short-lived
  ticket naming one workflow, for opening
  `GET /v1/workflows/{id}/events?ticket=...` with a browser `EventSource`
  (which cannot send `Authorization`). The long-lived token never goes in a
  URL; a ticket does, and it expires after `API_STREAM_TICKET_SECONDS` and
  opens nothing but that one workflow's stream, for nobody but the actor it
  was issued to. A ticket is only issued for a workflow the caller owns:
  another actor's is a 404. See `apps.api.auth`.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from apps.api.auth import ApiAuth, current_access, get_api_auth, require_principal, requires_scope
from apps.api.routes.events import get_progress_log
from personalos.domain.auth import AuthMethod, Principal, Scope
from personalos.domain.errors import NotFound
from personalos.domain.ownership import ActorScope
from personalos.persistence.workflow_progress import WorkflowProgressLog

router = APIRouter()


class WhoAmIResponse(BaseModel):
    """The verified caller: who, what kind of identity, and what its token allows."""

    actor_id: str
    method: AuthMethod
    kind: str
    scopes: list[Scope] = Field(default_factory=list)


class StreamTicketRequest(BaseModel):
    """Which workflow's event stream the ticket opens. The actor is the caller's."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: UUID


class StreamTicketResponse(BaseModel):
    """A ticket, when it expires, and the URL to open with it."""

    ticket: str
    expires_at: datetime
    stream_url: str


@router.get("/auth/whoami", response_model=WhoAmIResponse)
def whoami(principal: Principal = Depends(require_principal)) -> WhoAmIResponse:
    """The actor this caller's credential verified as."""
    return WhoAmIResponse(
        actor_id=principal.actor_id,
        method=principal.method,
        kind=principal.kind.value,
        scopes=sorted(principal.scopes, key=lambda scope: scope.value),  # type: ignore[arg-type]
    )


@router.post("/auth/stream-tickets", response_model=StreamTicketResponse, status_code=201)
@requires_scope(Scope.READ)
def create_stream_ticket(
    body: StreamTicketRequest,
    principal: Principal = Depends(require_principal),
    auth: ApiAuth = Depends(get_api_auth),
    log: WorkflowProgressLog = Depends(get_progress_log),
    access: ActorScope = Depends(current_access),
) -> StreamTicketResponse:
    """A short-lived ticket for one of the caller's workflows' event stream."""
    if not log.workflow_exists(body.workflow_id, access=access):
        raise NotFound(
            f"workflow '{body.workflow_id}' not found",
            details={"workflow_id": str(body.workflow_id)},
        )
    issued = auth.tickets.issue(principal, body.workflow_id)
    return StreamTicketResponse(
        ticket=issued.ticket,
        expires_at=issued.expires_at,
        stream_url=f"/v1/workflows/{body.workflow_id}/events?ticket={issued.ticket}",
    )


__all__ = ["router", "WhoAmIResponse", "StreamTicketRequest", "StreamTicketResponse"]

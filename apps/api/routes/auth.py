"""Who the caller is, and tickets for the event stream. Version 1.

- `GET /v1/auth/whoami` is the verified principal: the actor every route
  records for this caller, and how it was established. A UI calls it to check
  its token before showing anything.
- `POST /v1/auth/stream-tickets` exchanges the bearer token for a short-lived
  ticket naming one workflow, for opening
  `GET /v1/workflows/{id}/events?ticket=...` with a browser `EventSource`
  (which cannot send `Authorization`). The long-lived token never goes in a
  URL; a ticket does, and it expires after `API_STREAM_TICKET_SECONDS` and
  opens nothing but that one workflow's stream. See `apps.api.auth`.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from apps.api.auth import ApiAuth, get_api_auth, require_principal
from personalos.domain.auth import AuthMethod, Principal

router = APIRouter()


class WhoAmIResponse(BaseModel):
    """The verified caller."""

    actor_id: str
    method: AuthMethod


class StreamTicketRequest(BaseModel):
    """Which workflow's event stream the ticket opens."""

    workflow_id: UUID


class StreamTicketResponse(BaseModel):
    """A ticket, when it expires, and the URL to open with it."""

    ticket: str
    expires_at: datetime
    stream_url: str


@router.get("/auth/whoami", response_model=WhoAmIResponse)
def whoami(principal: Principal = Depends(require_principal)) -> WhoAmIResponse:
    """The actor this caller's credential verified as."""
    return WhoAmIResponse(actor_id=principal.actor_id, method=principal.method)


@router.post("/auth/stream-tickets", response_model=StreamTicketResponse, status_code=201)
def create_stream_ticket(
    body: StreamTicketRequest,
    principal: Principal = Depends(require_principal),
    auth: ApiAuth = Depends(get_api_auth),
) -> StreamTicketResponse:
    """A short-lived ticket for one workflow's event stream."""
    issued = auth.tickets.issue(principal, body.workflow_id)
    return StreamTicketResponse(
        ticket=issued.ticket,
        expires_at=issued.expires_at,
        stream_url=f"/v1/workflows/{body.workflow_id}/events?ticket={issued.ticket}",
    )


__all__ = ["router", "WhoAmIResponse", "StreamTicketRequest", "StreamTicketResponse"]

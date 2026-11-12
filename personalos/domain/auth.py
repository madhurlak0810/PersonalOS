"""Who is calling the API: the verified principal, and the credential shapes.

A request's actor is whatever its credential verified as -- never a header or
a body field the caller filled in. `Principal` is that verified identity; the
API derives it in `apps.api.auth` and every route takes its actor from there.

The credential shapes live here, not next to the code that mints them,
because `personalos.domain.redaction` matches on them too: a token is
redacted by shape even in a process that never loaded it (the CLI rotating it
while the API logs).
"""

from dataclasses import dataclass
from enum import Enum

#: Prefix of the long-lived local API token: `pos_at_<43 url-safe chars>`.
API_TOKEN_PREFIX = "pos_at_"

#: Prefix of a short-lived, workflow-scoped SSE stream ticket.
STREAM_TICKET_PREFIX = "pos_st_"


class AuthMethod(str, Enum):
    """How a principal was established. Recorded, never trusted for scoping."""

    #: The locally generated bearer token (`personalos.cli auth token`).
    LOCAL_TOKEN = "local_token"
    #: A short-lived ticket for one workflow's event stream.
    STREAM_TICKET = "stream_ticket"
    #: Authentication is turned off (`API_AUTH_ENABLED=false`, loopback only).
    DISABLED = "disabled"


@dataclass(frozen=True)
class Principal:
    """A verified caller.

    `actor_id` is what routes record as the actor: who decided an approval,
    whose profile is read. Which actor may do what is the next phase's
    authorization model; this phase only guarantees the actor is not
    self-declared.
    """

    actor_id: str
    method: AuthMethod

    def __repr__(self) -> str:
        return f"Principal(actor_id={self.actor_id!r}, method={self.method.value})"


__all__ = ["API_TOKEN_PREFIX", "STREAM_TICKET_PREFIX", "AuthMethod", "Principal"]

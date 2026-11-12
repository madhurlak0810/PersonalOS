"""API authentication: who a request is from, established before any route runs.

**Every route but `/health` requires it.** `create_app` mounts each router
with `require_principal` as a dependency, and `tests/unit/test_api_auth.py`
walks every registered route to prove none was missed. A route that needs the
caller's identity takes `actor_id: str = Depends(current_actor)`: the actor is
whatever the credential verified as, never a header or body field.

**The mechanism is pluggable.** `require_principal` only knows the
`TokenVerifier` protocol -- a bearer string in, a `Principal` or `None` out.
The MVP verifier is `LocalTokenVerifier`, over the one locally generated token
(`personalos.secrets.api_token`). An OIDC verifier (validate a JWT against an
issuer's JWKS, map `sub` to an actor) implements the same `verify` and is
installed by `build_api_auth`; no route changes.

**The event stream authenticates too.** A browser `EventSource` cannot set
`Authorization`, and the long-lived token must not go in a URL (URLs land in
access logs, browser history and `Referer`). So
`GET /v1/workflows/{id}/events` accepts either the bearer header (a
fetch-based SSE client) or `?ticket=`: a stream ticket from
`POST /v1/auth/stream-tickets`, which is HMAC-signed, names one workflow and
one actor, and expires after `API_STREAM_TICKET_SECONDS`. Tickets are
stateless, so any API worker process can check one another issued; the
signing key is derived from the local token, so rotating the token voids
every outstanding ticket.

**Turning it off is loopback-only.** `API_AUTH_ENABLED=false` makes every
request the configured `API_ACTOR`, and `ensure_safe_bind` refuses to start
such an API on anything but a loopback address.
"""

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import UUID

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from personalos.config import settings
from personalos.domain.auth import STREAM_TICKET_PREFIX, AuthMethod, Principal
from personalos.domain.errors import Unauthenticated
from personalos.secrets.api_token import LocalApiToken

logger = logging.getLogger(__name__)

#: How often `LocalTokenVerifier` re-reads the stored token, so a rotation by
#: `auth token --rotate` takes effect in a running API without a restart.
DEFAULT_TOKEN_RELOAD_SECONDS = 5.0

_TICKET_KEY_CONTEXT = b"personalos stream ticket v1"


# --- Verification ---------------------------------------------------------------


@runtime_checkable
class TokenVerifier(Protocol):
    """Turns a bearer credential into a principal. The seam OIDC replaces."""

    def verify(self, token: str) -> Principal | None:
        """The principal `token` authenticates, or `None` if it does not verify."""
        ...


class LocalTokenVerifier:
    """Verifies the locally generated token, re-reading it every few seconds."""

    def __init__(
        self,
        token: LocalApiToken,
        *,
        actor_id: str,
        reload_seconds: float = DEFAULT_TOKEN_RELOAD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Verify against the token `token` manages, as `actor_id`."""
        self._token = token
        self._actor_id = actor_id
        self._reload_seconds = reload_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: bytes | None = None
        self._loaded_at: float | None = None

    def ensure_token(self) -> bool:
        """Mint the token if there is none yet. Returns whether one was created."""
        _, created = self._token.load_or_create()
        with self._lock:
            self._loaded_at = None
        return created

    def verify(self, token: str) -> Principal | None:
        """A constant-time comparison with the stored token."""
        current = self._current()
        if current is None or not token:
            return None
        if not hmac.compare_digest(token.encode("utf-8"), current):
            return None
        return Principal(actor_id=self._actor_id, method=AuthMethod.LOCAL_TOKEN)

    def ticket_key(self) -> bytes | None:
        """The stream-ticket signing key, derived from the current token."""
        current = self._current()
        if current is None:
            return None
        return hmac.new(current, _TICKET_KEY_CONTEXT, hashlib.sha256).digest()

    def _current(self) -> bytes | None:
        with self._lock:
            now = self._clock()
            stale = self._loaded_at is None or now - self._loaded_at >= self._reload_seconds
            if stale:
                stored = self._token.current()
                self._cached = stored.reveal().encode("utf-8") if stored else None
                self._loaded_at = now
            return self._cached


# --- Stream tickets -------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True)
class StreamTicket:
    """A ticket as issued: the opaque string and when it stops working."""

    ticket: str
    expires_at: datetime


class StreamTicketSigner:
    """Issues and checks short-lived, workflow-scoped stream tickets.

    A ticket is `pos_st_<payload>.<mac>`: the payload names the actor, the
    workflow and the expiry; the MAC is HMAC-SHA256 under `key()`. Nothing is
    stored, so a ticket works in whichever API worker the stream lands on.
    A ticket is reusable until it expires, which is what lets `EventSource`
    reconnect on its own after a dropped connection.
    """

    def __init__(
        self,
        key: Callable[[], bytes | None],
        *,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ):
        """Sign with whatever `key()` returns at the time; `None` means no tickets."""
        self._key = key
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    def issue(self, principal: Principal, workflow_id: UUID) -> StreamTicket:
        """A ticket for `principal` to read `workflow_id`'s events."""
        key = self._key()
        if key is None:
            raise Unauthenticated("stream tickets are unavailable: no API token is configured")
        expires = int(self._clock() + self.ttl_seconds)
        body = json.dumps(
            {"a": principal.actor_id, "w": str(workflow_id), "e": expires},
            separators=(",", ":"),
        ).encode("utf-8")
        payload = _b64(body)
        mac = _b64(hmac.new(key, payload.encode("ascii"), hashlib.sha256).digest())
        return StreamTicket(
            ticket=f"{STREAM_TICKET_PREFIX}{payload}.{mac}",
            expires_at=datetime.fromtimestamp(expires, tz=timezone.utc),
        )

    def verify(self, ticket: str, workflow_id: UUID) -> Principal | None:
        """The principal a ticket was issued to, if it is genuine, current and for this workflow."""
        key = self._key()
        if key is None or not ticket.startswith(STREAM_TICKET_PREFIX):
            return None
        payload, dot, mac = ticket[len(STREAM_TICKET_PREFIX) :].partition(".")
        if not dot:
            return None
        expected = _b64(hmac.new(key, payload.encode("ascii", "replace"), hashlib.sha256).digest())
        if not hmac.compare_digest(mac.encode("ascii", "replace"), expected.encode("ascii")):
            return None
        try:
            claims = json.loads(_unb64(payload))
            actor, scoped, expires = str(claims["a"]), str(claims["w"]), int(claims["e"])
        except (ValueError, KeyError, TypeError):
            return None
        if scoped != str(workflow_id) or self._clock() >= expires:
            return None
        return Principal(actor_id=actor, method=AuthMethod.STREAM_TICKET)


# --- What the app holds ----------------------------------------------------------


@dataclass
class ApiAuth:
    """The API's authentication: on or off, the verifier, the ticket signer."""

    verifier: TokenVerifier
    tickets: StreamTicketSigner
    enabled: bool = True
    #: The actor every request is when `enabled` is false.
    disabled_actor: str = "api"


def build_api_auth() -> ApiAuth:
    """Authentication as configured: the local token unless it is turned off."""
    from personalos.bootstrap import build_local_api_token

    ttl = settings.api_stream_ticket_seconds
    if not settings.api_auth_enabled:
        process_key = secrets.token_bytes(32)
        return ApiAuth(
            verifier=_RejectAll(),
            tickets=StreamTicketSigner(lambda: process_key, ttl_seconds=ttl),
            enabled=False,
            disabled_actor=settings.api_actor,
        )
    verifier = LocalTokenVerifier(build_local_api_token(), actor_id=settings.api_actor)
    return ApiAuth(
        verifier=verifier,
        tickets=StreamTicketSigner(verifier.ticket_key, ttl_seconds=ttl),
    )


class _RejectAll:
    def verify(self, token: str) -> Principal | None:
        return None


# --- FastAPI dependencies --------------------------------------------------------

#: Reads `Authorization: Bearer <token>`. `auto_error=False` so a missing or
#: non-bearer header reaches `require_principal` and becomes the same 401
#: envelope as a wrong token. Also what puts the "Authorize" button in /docs.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="The local API token: `python -m personalos.cli auth token`.",
)


def get_api_auth(request: Request) -> ApiAuth:
    """The app's `ApiAuth`, built from settings the first time it is needed."""
    auth = getattr(request.app.state, "api_auth", None)
    if auth is None:
        auth = build_api_auth()
        request.app.state.api_auth = auth
    return auth


def _authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
    auth: ApiAuth,
) -> Principal:
    principal: Principal | None
    if not auth.enabled:
        principal = Principal(actor_id=auth.disabled_actor, method=AuthMethod.DISABLED)
    else:
        principal = auth.verifier.verify(credentials.credentials) if credentials else None
    if principal is None:
        raise Unauthenticated()
    request.state.principal = principal
    return principal


def require_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth: ApiAuth = Depends(get_api_auth),
) -> Principal:
    """FastAPI dependency: the verified caller, or a 401."""
    return _authenticate(request, credentials, auth)


def current_actor(principal: Principal = Depends(require_principal)) -> str:
    """FastAPI dependency: the verified caller's actor id."""
    return principal.actor_id


def require_stream_principal(
    workflow_id: UUID,
    request: Request,
    ticket: str | None = Query(
        default=None,
        description="A stream ticket from `POST /v1/auth/stream-tickets`, for `EventSource`.",
    ),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth: ApiAuth = Depends(get_api_auth),
) -> Principal:
    """FastAPI dependency for an event stream: a bearer token, or a ticket for this workflow."""
    if not auth.enabled or credentials is not None or not ticket:
        return _authenticate(request, credentials, auth)
    principal = auth.tickets.verify(ticket, workflow_id)
    if principal is None:
        raise Unauthenticated()
    request.state.principal = principal
    return principal


# --- Bind address ----------------------------------------------------------------


class UnsafeBindAddress(RuntimeError):
    """The API was asked to listen beyond loopback with authentication off."""


def is_loopback_host(host: str) -> bool:
    """Whether `host` is a loopback address. Any name but `localhost` counts as not."""
    candidate = host.strip().strip("[]")
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def ensure_safe_bind(host: str, *, auth_enabled: bool) -> None:
    """Refuse a non-loopback bind address when authentication is off."""
    if not auth_enabled and not is_loopback_host(host):
        raise UnsafeBindAddress(
            f"refusing to listen on {host!r} with API_AUTH_ENABLED=false: anyone who can "
            f"reach it could act as {settings.api_actor!r}. Bind 127.0.0.1, or enable auth."
        )


__all__ = [
    "ApiAuth",
    "LocalTokenVerifier",
    "StreamTicket",
    "StreamTicketSigner",
    "TokenVerifier",
    "UnsafeBindAddress",
    "bearer_scheme",
    "build_api_auth",
    "current_actor",
    "ensure_safe_bind",
    "get_api_auth",
    "is_loopback_host",
    "require_principal",
    "require_stream_principal",
]

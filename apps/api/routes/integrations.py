"""Connected integrations over HTTP: status, Google consent, disconnect. Version 1.

- `GET /v1/integrations` lists each integration: whether it is connected, as
  which account, with which scopes, until when, and whether it needs
  reconnecting. Google comes first; after it, every other integration the
  worker has health-checked (`gmail`, `google_calendar`, `google_mcp`, each
  job board, the embedder, each local model server), with its health
  (`healthy`, `degraded` or `down`), the reason, `last_success_at`, the last
  error (redacted) and the `action` that fixes it: `reconnect` or `retry`.
  It works before anything is configured.
- `POST /v1/integrations/google/connect` begins a consent flow and returns the
  `authorization_url` to open in a browser. With no body it is read-only;
  `{"enable": ["gmail.send"]}` asks for a write capability, and is refused
  (409) until a read-only connection exists -- write consent is its own step.
- `GET /v1/integrations/google/callback` is where Google redirects the
  browser. It is on `127.0.0.1`, so the API must be running on this machine.
  It stores the refresh token in the OS keychain and answers with the
  connection.
- `POST /v1/integrations/google/revoke` revokes the grant at Google and
  deletes the keychain entry.

No response here has, or could have, a field holding a token: every body is a
`personalos.domain.integrations` model. The callback's query string carries
the authorization code, and the access log line for it goes through the same
log redaction as everything else.

Behind the API token, except the callback, and deployment-wide: there is one
set of Google connections per PersonalOS install. **The callback is the one
exemption besides `/health`**, because it is Google redirecting the user's
browser, which has no bearer token to send. It is authenticated instead by
its `state`: single-use, minted only by an authenticated `connect`, bound to
the PKCE verifier held server-side, and expiring with the pending consent. A
request with any other `state` is refused before anything is stored. It is
mounted from `callback_router` so `apps.api.main` can leave it out of the
bearer dependency, and `tests/unit/test_api_auth.py` names it explicitly.
"""

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from personalos.bootstrap import build_google_connection_reader, build_google_connector
from personalos.domain.failures import IntegrationHealth
from personalos.domain.health import GOOGLE, GOOGLE_INTEGRATIONS
from personalos.domain.integrations import (
    ConsentRequest,
    GoogleCapability,
    GoogleConnection,
    IntegrationStatus,
    RevokeResult,
)
from personalos.executor.google_connect import GoogleConnectionReader, GoogleConnector
from personalos.persistence.database import SessionLocal
from personalos.secrets.oauth_loopback import LOOPBACK_HOST

router = APIRouter()

#: Name of the callback route, for building its URL.
CALLBACK_ROUTE = "google_oauth_callback"


class IntegrationsResponse(BaseModel):
    """Every integration's connection state."""

    integrations: list[IntegrationStatus]


class GoogleConnectRequest(BaseModel):
    """Which capabilities beyond read-only to ask consent for. Empty is read-only."""

    model_config = ConfigDict(extra="forbid")

    enable: list[GoogleCapability] = Field(default_factory=list)


class GoogleRevokeRequest(BaseModel):
    """Which account to disconnect; every connected one when absent."""

    model_config = ConfigDict(extra="forbid")

    account: str | None = Field(default=None, max_length=200)


class GoogleRevokeResponse(BaseModel):
    """What was disconnected."""

    results: list[RevokeResult]


def get_google_connection_reader() -> GoogleConnectionReader:
    """FastAPI dependency: reads connections; needs no keychain or OAuth client."""
    return build_google_connection_reader(SessionLocal)


def get_google_connector() -> GoogleConnector:
    """FastAPI dependency: the consent flow, over the keychain and the OAuth client.

    A 503 (`PortNotConfigured`) until the OAuth client is set up.
    """
    return build_google_connector(SessionLocal)


def _callback_uri(request: Request) -> str:
    """The callback's URL on the loopback address, whatever host the caller used."""
    url = request.url_for(CALLBACK_ROUTE)
    return str(url.replace(scheme="http", hostname=LOOPBACK_HOST))


@router.get("/integrations", response_model=IntegrationsResponse)
def list_integrations(reader: GoogleConnectionReader = Depends(get_google_connection_reader)):
    """Every integration's connection and health state. Never includes token material."""
    google = reader.status()
    others = [
        _status(health, google) for health in reader.health.all() if health.provider != GOOGLE
    ]
    return IntegrationsResponse(integrations=[google, *others])


def _status(health: IntegrationHealth, google: IntegrationStatus) -> IntegrationStatus:
    """One health-checked integration, as `GET /v1/integrations` shows it.

    A Google integration is connected when the Google account is; any other
    is configured, which is what having been checked at all means.
    """
    connected = google.connected if health.provider in GOOGLE_INTEGRATIONS else True
    return IntegrationStatus(
        provider=health.provider,
        connected=connected,
        reconnect_required=health.reconnect_required,
        health=health.status,
        health_reason=health.reason,
        action=health.action,
        last_error=health.last_error,
        last_success_at=health.last_success_at,
        last_checked_at=health.last_checked_at,
    )


@router.post("/integrations/google/connect", response_model=ConsentRequest, status_code=201)
def begin_google_connect(
    request: Request,
    body: GoogleConnectRequest | None = None,
    connector: GoogleConnector = Depends(get_google_connector),
):
    """Begin Google consent; open the returned `authorization_url` in a browser."""
    enable = body.enable if body else []
    return connector.begin(redirect_uri=_callback_uri(request), enable=enable)


#: Routes authenticated by something other than the bearer token. Only the
#: OAuth callback, whose `state` is its credential.
callback_router = APIRouter()


@callback_router.get(
    "/integrations/google/callback",
    response_model=GoogleConnection,
    name=CALLBACK_ROUTE,
)
async def google_oauth_callback(
    state: str | None = Query(default=None, max_length=64),
    code: str | None = Query(default=None, max_length=2048),
    error: str | None = Query(default=None, max_length=64),
    connector: GoogleConnector = Depends(get_google_connector),
):
    """Google's redirect: finish the consent and store the refresh token."""
    return await connector.complete(state=state, code=code, error=error)


@router.post("/integrations/google/revoke", response_model=GoogleRevokeResponse)
async def revoke_google(
    body: GoogleRevokeRequest | None = None,
    connector: GoogleConnector = Depends(get_google_connector),
):
    """Revoke at Google and delete the keychain entry."""
    results = await connector.revoke(body.account if body else None)
    return GoogleRevokeResponse(results=results)


__all__ = [
    "router",
    "callback_router",
    "get_google_connection_reader",
    "get_google_connector",
    "IntegrationsResponse",
    "GoogleConnectRequest",
    "GoogleRevokeRequest",
    "GoogleRevokeResponse",
]
